"""Inference."""

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

from models import model
from layers.data_layer import get_images

#import utils.locality_aware_nms as nms_locality
from utils import lanms
from utils.config import FLAGS
from utils.polygon_transform import restore_poly


def resize_image(img, max_side_len=512, max_stride=32):
    """Resize image with max stride."""
    height, width, _ = img.shape

    ratio = float(max_side_len) / max(height, width)
    resize_h = int(height * ratio)
    resize_w = int(width * ratio)

    # ensure the target size to be N * max_stride
    resize_h = int((resize_h + max_stride - 1) / max_stride) * max_stride
    resize_w = int((resize_w + max_stride - 1) / max_stride) * max_stride
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(height)
    ratio_w = resize_w / float(width)

    return img, (ratio_h, ratio_w)


def resize_image_old(img, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param img: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = img.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return img, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.5, box_thresh=0.1, nms_thres=0.2):
    """
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, :]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_poly(4 * xy_text[:, ::-1],
                                     geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N * 4 * 2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def refine_boxes_by_border_pts(boxes, score_map, geo_map, score_map_thresh=0.5):
    """Refine box by border points."""
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, :]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)[:, ::-1] # (m, 2)
    if len(xy_text) == 0:
        return boxes

    boxes_refine = boxes.copy() # (m, n, 2)
    for i, box in enumerate(boxes):
        # find left/right border center pt
        lb_center_pt = np.array([(box[0, 0] + box[3, 0]) / 2.0,
                                 (box[0, 1] + box[3, 1]) / 2.0])
        rb_center_pt = np.array([(box[1, 0] + box[2, 0]) / 2.0,
                                 (box[1, 1] + box[2, 1]) / 2.0])
        lens = [None, None, None, None]
        for j in range(4):
            lens[j] = np.linalg.norm(box[(j + 1) % 4] - box[j])
        l_shrink_len = min(0.1 * (lens[0] + lens[2]) / 2.0, 0.5 * lens[3])
        r_shrink_len = min(0.1 * (lens[0] + lens[2]) / 2.0, 0.5 * lens[1])
        lrb_center_len = np.linalg.norm(rb_center_pt - lb_center_pt)

        lb_center_pt_shrink = lb_center_pt + (l_shrink_len / lrb_center_len) * \
                              (rb_center_pt - lb_center_pt)
        rb_center_pt_shrink = rb_center_pt + (r_shrink_len / lrb_center_len) * \
                              (lb_center_pt - rb_center_pt)

        # refine left border
        # find left border points
        lb_delta_xy = 4 * xy_text - np.tile(lb_center_pt_shrink[np.newaxis, :], (xy_text.shape[0], 1))
        lb_dist = np.linalg.norm(lb_delta_xy, axis=1)
        lb_xy_text = xy_text[lb_dist < l_shrink_len, :] # (m2, 2)
        # refine
        if len(lb_xy_text) > 0:
            xy_geo = geo_map[lb_xy_text[:, 1], lb_xy_text[:, 0], :] # (m2, 8)
            lb_pred_pts = 4 * np.tile(lb_xy_text, (1, 2)) - xy_geo[:, (-2, -1, 0, 1)]
            boxes_refine[i, (-1, 0), :] = np.mean(lb_pred_pts, axis=0).reshape(2, 2)

        # refine right border
        # find right border points
        rb_delta_xy = 4 * xy_text - np.tile(rb_center_pt_shrink[np.newaxis, :], (xy_text.shape[0], 1))
        rb_dist = np.linalg.norm(rb_delta_xy, axis=1)
        rb_xy_text = xy_text[rb_dist < r_shrink_len, :] # (m2, 2)
        # refine
        if len(rb_xy_text) > 0:
            xy_geo = geo_map[rb_xy_text[:, 1], rb_xy_text[:, 0], :] # (m2, 8)
            rb_pred_pts = 4 * np.tile(rb_xy_text, (1, 2)) - xy_geo[:, 2:6]
            boxes_refine[i, 1:3, :] = np.mean(rb_pred_pts, axis=0).reshape(2, 2)

    return boxes_refine


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                      trainable=False)

        f_shared = model.resnet50_4unet32(input_images, is_training=False)
        f_score, f_geo = model.det_head(f_shared, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            model_path = os.path.join(FLAGS.checkpoint_dir,
                                      os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            # save infer graph
            if FLAGS.save_infer_graph:
                saver.save(sess, '{}/model.ckpt'.format(FLAGS.output_dir), global_step=global_step)

            img_fn_list = get_images(FLAGS.test_data_dir)
            for img_fn in img_fn_list:
                img = cv2.imread(img_fn)[:, :, ::-1]
                start_time = time.time()
                img_resized, (ratio_h, ratio_w) = resize_image(img, max_side_len=FLAGS.max_side_len)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geo = sess.run([f_score, f_geo],
                                      feed_dict={input_images: [img_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geo, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    img_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    # box refinement
                    if FLAGS.enable_box_refinement:
                        boxes = refine_boxes_by_border_pts(boxes, score, geo)
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                img_basename = os.path.basename(img_fn)
                img_prefix = img_basename[:img_basename.rfind('.')]

                # save score map
                if FLAGS.save_score_map:
                    dst_h = int(4 * score[0].shape[0] / ratio_h)
                    dst_w = int(4 * score[0].shape[1] / ratio_w)
                    score_resize = cv2.resize(score[0], dsize=(dst_w, dst_h),
                                              interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite('{}/{}_score.jpg'.format(FLAGS.output_dir, img_prefix),
                                score_resize * 255)

                # save to file
                if boxes is not None:
                    res_file = os.path.join(FLAGS.output_dir, '{}.txt'.format(img_prefix))
                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            #box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or \
                               np.linalg.norm(box[3] - box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                                box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))],
                                          True, color=(255, 255, 0), thickness=2)
                            for point_idx in range(4):
                                cv2.putText(img[:, :, ::-1], str(point_idx),
                                            (box[point_idx, 0], box[point_idx, 1]),
                                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(img_fn))
                    cv2.imwrite(img_path, img[:, :, ::-1])

if __name__ == '__main__':
    tf.app.run()