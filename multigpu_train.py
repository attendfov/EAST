"""Multigpu train."""

import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from models import model
from layers import data_layer
from utils.config import FLAGS


gpus = list(range(len(FLAGS.gpu_list.split(','))))


def tower_loss(imgs, score_maps, geo_maps, training_masks, reuse_variables=None):
    """Tower loss."""
    # Build graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_shared = model.resnet50_4unet32(imgs, is_training=True)
        f_score, f_geo = model.det_head(f_shared, is_training=True)

    cls_loss, geo_loss = model.det_loss(score_maps, f_score,
                                        geo_maps, f_geo,
                                        training_masks)
    cls_lw, geo_lw = 1.0, 1.0
    model_loss = cls_lw * cls_loss + geo_lw * geo_loss
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Add summary
    if reuse_variables is None:
        tf.summary.image('input', imgs)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geo[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, cls_loss, geo_loss, cls_lw, geo_lw


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MkDir(FLAGS.checkpoint_dir)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
            tf.gfile.MkDir(FLAGS.checkpoint_dir)

    # Define input placeholders
    input_imgs = tf.placeholder(tf.float32,
                                shape=[None, None, None, 3],
                                name='input_imgs')
    input_score_maps = tf.placeholder(tf.float32,
                                      shape=[None, None, None, 1],
                                      name='input_score_maps')
    # 8 channels for coordinate offsets and 1 channel for short_edge_norm
    input_geo_maps = tf.placeholder(tf.float32,
                                    shape=[None, None, None, 9],
                                    name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32,
                                          shape=[None, None, None, 1],
                                          name='input_training_masks')

    # Define optimizer
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                               decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    # Add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # Split
    input_imgs_split = tf.split(input_imgs, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_imgs_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss, cls_loss, geo_loss, cls_lw, geo_lw = tower_loss(iis,
                        isms, igms, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                # Clip gradients
                grads_zip, params = zip(*grads)
                grads_clip, _ = tf.clip_by_global_norm(grads_zip, 10.0)
                grads = zip(grads_clip, params)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add summary
    tf.summary.scalar("total_gradient_norm", tf.global_norm(grads))
    summary_op = tf.summary.merge_all()

    # Save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Batch norm updates
    with tf.control_dependencies([variables_averages_op,
                                  apply_gradient_op,
                                  batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    # Saver and writer
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=30)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, tf.get_default_graph())

    init = tf.global_variables_initializer()

    # Load pretrain or previous model
    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                slim.get_trainable_variables(), ignore_missing_vars=True)

    # Open a session and training
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = False
    with tf.Session(config=config) as sess:
        # Load ro initialize the model
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        # Begin training
        data_generator = data_layer.get_batch(num_workers=FLAGS.num_readers,
                                              input_size=FLAGS.input_size,
                                              batch_size=FLAGS.batch_size_per_gpu * len(gpus),
                                              background_ratio=1.0 / 8)

        start = time.time()
        for step in range(FLAGS.max_steps):
            # Get a batch of data
            data = next(data_generator)

            # Define input_feed and output_feed
            input_feed = {input_imgs: data[0],
                          input_score_maps: data[2],
                          input_geo_maps: data[3],
                          input_training_masks: data[4]}
            output_feed = [total_loss, model_loss, cls_loss, geo_loss,
                           summary_op, train_op, learning_rate]

            tl, ml, cl, gl, _, _, lr = sess.run(output_feed, feed_dict=input_feed)
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10.0
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus)) / \
                                          (time.time() - start)
                start = time.time()
                sys_time = datetime.now().isoformat()
                print('{}] Step {:07d}, lr = {:.7f}, total_loss = {:.4f}, model_loss = {:.4f}'.format(\
                      sys_time, step, lr, tl, ml) + ', {:.2f} seconds/step'.format(avg_time_per_step))
                print('{}]     cls_loss = {:.4f} (* {:.2f} = {:.4f})'.format(\
                      sys_time, cl, cls_lw, cl * cls_lw))
                print('{}]     geo_loss = {:.4f} (* {:.2f} = {:.4f})'.format(\
                      sys_time, gl, geo_lw, gl * geo_lw))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                sum_output_feed = [total_loss, model_loss, summary_op, train_op]
                tl, ml, sum_str, _ = sess.run(sum_output_feed, feed_dict=input_feed)
                summary_writer.add_summary(sum_str, global_step=step)

if __name__ == '__main__':
    tf.app.run()