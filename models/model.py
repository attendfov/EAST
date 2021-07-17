import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim
from nets import resnet_v1
from utils.config import FLAGS


# ---------------------------------------------------------------------
# Useful layers
# ---------------------------------------------------------------------

def unpool(inputs):
    return tf.image.resize_bilinear(inputs,
                                    size=[tf.shape(inputs)[1] * 2,
                                    tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    """
    image normalization
    :param images:
    :param means:
    :return:
    """
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


# ---------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------

def resnet50_4unet32(images, weight_decay=1e-5, is_training=True):
    """
    define the model, we use slim's implemention of resnet
    """
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

    return g[3]


# ---------------------------------------------------------------------
# Multi-task heads
# ---------------------------------------------------------------------

def det_head(f_shared, weight_decay=1e-5, is_training=True):
    """Detection head."""
    with tf.variable_scope('det_head'):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            # score branch
            # here we use a slightly different way for regression part,
            # we use a sigmoid to limit the regression range
            f_score = slim.conv2d(f_shared, 1, 1,
                                  activation_fn=tf.nn.sigmoid,
                                  normalizer_fn=None)

            # geo branch
            # 8 channel of coordinate offsets
            # offset is between [-input_size, input_size]
            f_geo = (slim.conv2d(f_shared, 8, 1,
                                 activation_fn=tf.nn.sigmoid,
                                 normalizer_fn=None) - 0.5) * 2 * FLAGS.input_size

    return f_score, f_geo


# ---------------------------------------------------------------------
# Loss layers
# ---------------------------------------------------------------------

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + \
            tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def smooth_l1_loss(y_true_geo, y_pred_geo,
                   y_true_cls, training_mask):
    channels = 8
    y_true_geo_split = tf.split(value=y_true_geo, num_or_size_splits=(channels + 1), axis=3)
    short_edge_norm = y_true_geo_split[-1]
    offset_gt = y_true_geo_split[:channels]

    offset_pred = tf.split(value=y_pred_geo, num_or_size_splits=channels, axis=3)

    L_g = 0
    for i in range(channels):
        offset_diff = offset_gt[i] - offset_pred[i]
        abs_offset_diff = tf.abs(offset_diff)
        # stop backward gradient to tensor smooth_l1_sign
        smooth_l1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_offset_diff, 1.0)))

        in_loss_offset = 0.5 * tf.pow(offset_diff, 2) * smooth_l1_sign + \
                         (abs_offset_diff - 0.5) * (1 - smooth_l1_sign)
        out_loss_offset = (short_edge_norm / channels) * in_loss_offset
        L_g += out_loss_offset

    loss = tf.reduce_sum(L_g * y_true_cls * training_mask) / (tf.reduce_sum(y_true_cls * training_mask) + 1e-5)

    tf.summary.scalar('geometry_QUAD', loss)

    return loss


def det_loss(y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
    """
    define the detection loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the smooth l1 loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    """
    cls_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    geo_loss = smooth_l1_loss(y_true_geo, y_pred_geo, y_true_cls, training_mask)

    return cls_loss, geo_loss