# coding:utf-8

"""Data layer."""

import glob
import csv
import cv2
import time
import os
import numpy as np
import xml.etree.ElementTree as ET

import tensorflow as tf

from utils.data_util import GeneratorEnqueuer
from utils.config import FLAGS


def get_images(data_dir):
    """
    find image files in data dir
    :return: list of files found
    """
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for ext in exts:
        files.extend(glob.glob('{}/*.{}'.format(data_dir, ext)))
    print('Find {} images in {}'.format(len(files), data_dir))
    return files


def get_images_by_list(img_list_path):
    """Get images based on img_list_path."""
    fid = open(img_list_path, 'r')
    files = [line.strip() for line in fid.readlines()]
    print('Find {} images in {}'.format(len(files), img_list_path))
    return files


def get_labels_by_list(xml_list_path):
    """Get labels based on xml_list_path."""
    fid = open(xml_list_path, 'r')
    files = [line.strip() for line in fid.readlines()]
    print('Find {} labels in {}'.format(len(files), xml_list_path))
    return files


def load_annoataion(p):
    """
    load annotation from the text file
    :param p:
    :return:
    """
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def permute(ori_poly):
    poly = np.array(ori_poly, dtype=np.float32)
    num_pt = poly.shape[0]
    p = np.zeros((num_pt, 2), dtype=np.float32)
    ind = np.argsort(poly[:, 0])[:2]
    pt_1 = ind[np.argmin(poly[ind, 1])]
    for i in range(num_pt):
        p[i] = poly[(i + pt_1) % num_pt]

    return p.tolist()


def compute_rotate_angle(poly):
    """Compute rotate angle."""
    # NOTE: only support quadrangle
    angle = np.arctan2(poly[1][1] - poly[0][1], poly[1][0] - poly[0][0])
    angle = 180.0 * angle / np.pi
    return angle


def parse_xml(xml_file):
    """Parse xml file."""
    text_polys, text_tags = [], []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root.findall('object_polygon'):
        poly = []
        for point in child.find('points').findall('point'):
            x = float(point.find('x').text)
            y = float(point.find('y').text)
            poly.append([x, y])

        if FLAGS.permute_points:
            poly = permute(poly)

        text_polys.append(poly)
        label = child.find('text').text
        tag = False
        if label == '*' or label == '###':
            tag = True
        # ignore large rotation text poly
        if FLAGS.ignore_large_rotate_poly:
            if abs(compute_rotate_angle(poly)) > 15:
                tag = True

        text_tags.append(tag)

    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def polygon_area(poly):
    """
    compute area of a polygon
    :param poly:
    :return:
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    """
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        # compute polygon area
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print(poly)
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def random_rotate_4dir(img, text_polys):
    """Randomly rotate image in four directions."""
    h, w, _ = img.shape
    text_polys_rot = np.zeros_like(text_polys)

    rd_value = np.random.random_sample()
    if rd_value < 5.0 / 8:
        # rotate 0
        img_rot = img.copy()
        text_polys_rot = text_polys.copy()
    elif rd_value < 6.0 / 8:
        # rotate 90
        img_rot = np.transpose(img, (1, 0, 2))
        img_rot = np.flip(img_rot, axis=0)  # flip upper-down
        text_polys_rot[:, :, 0] = text_polys[:, :, 1]
        text_polys_rot[:, :, 1] = w - text_polys[:, :, 0]
    elif rd_value < 7.0 / 8:
        # rotate 180
        img_rot = np.flip(img, axis=0)  # flip upper-down
        img_rot = np.flip(img_rot, axis=1)  # flip left-right
        text_polys_rot[:, :, 0] = w - text_polys[:, :, 0]
        text_polys_rot[:, :, 1] = h - text_polys[:, :, 1]
    else:
        # rotate 270
        img_rot = np.transpose(img, (1, 0, 2))
        img_rot = np.flip(img_rot, axis=1)  # flip left-right
        text_polys_rot[:, :, 0] = h - text_polys[:, :, 1]
        text_polys_rot[:, :, 1] = text_polys[:, :, 0]

    return img_rot, text_polys_rot


def crop_area(img, polys, tags, crop_background=False, max_tries=50):
    """
    make random crop from the input image
    :param img:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    """
    h, w, _ = img.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    # print(polys)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return img, polys, tags

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return img[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        img = img[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return img, polys, tags

    return img, polys, tags


def shrink_poly(poly, r):
    """
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    """
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print(poly)
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def generate_poly(img_size, polys, tags):
    """Generate polygon map."""
    point_num = int(polys.shape[1])  # (m, n, 2)
    h, w = img_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    xy_map = np.zeros((2 * point_num, h, w), dtype=np.float32)
    poly_map = np.zeros((2 * point_num + 1, h, w), dtype=np.float32)
    # (x1, y1, x2, y2, ..., xn, yn, short_edge_norm)
    geo_map = np.zeros((2 * point_num + 1, h, w), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)

    # xy_map
    xy_map[::2] = np.tile(np.arange(0, w)[np.newaxis, np.newaxis, :], (point_num, h, 1))
    xy_map[1::2] = np.tile(np.arange(0, h)[np.newaxis, :, np.newaxis], (point_num, 1, w))

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        r = [None, None, None, None]
        for i in range(point_num):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % point_num]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % point_num]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, 1)

        # if the poly is too small, then ignore it during training
        poly_h = 0.5 * (np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]))
        poly_w = 0.5 * (np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]))
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        # TODO(zuming): add FLAGS.max_text_size
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            tags[poly_idx] = True

        for point_idx in range(point_num):
            cv2.fillPoly(poly_map[2 * point_idx], shrinked_poly,
                         float(min(max(poly[point_idx, 0], 0), w)))
            cv2.fillPoly(poly_map[2 * point_idx + 1], shrinked_poly,
                         float(min(max(poly[point_idx, 1], 0), h)))
        # short_edge_norm
        cv2.fillPoly(poly_map[-1], shrinked_poly, 1.0 / max(min(poly_h, poly_w), 1.0))

    # geo_map
    geo_map[:(2 * point_num)] = (xy_map - poly_map[:(2 * point_num)]) * poly_mask
    geo_map[2 * point_num] = poly_map[-1] * poly_mask
    geo_map = np.transpose(geo_map, (1, 2, 0))

    return score_map, geo_map, training_mask, tags


def generator(input_size=512, batch_size=32,
              background_ratio=3. / 8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    """Generate data."""
    img_list = np.array(get_images_by_list(FLAGS.train_img_list_path))
    xml_list = np.array(get_labels_by_list(FLAGS.train_xml_list_path))
    assert (len(img_list) == len(xml_list))
    index = np.arange(0, img_list.shape[0])
    while True:
        np.random.shuffle(index)
        imgs = []
        img_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index:
            try:
                img_fn = img_list[i]
                img = cv2.imread(img_fn)
                h, w, _ = img.shape

                xml_fn = xml_list[i]
                if not os.path.exists(xml_fn):
                    print('xml file {} does not exists'.format(xml_fn))
                    continue

                # assert img_fn and txt_fn have the same prefix
                img_basename = os.path.basename(img_fn)
                img_prefix = img_basename[:img_basename.rfind('.')]
                xml_basename = os.path.basename(xml_fn)
                xml_prefix = xml_basename[:xml_basename.rfind('.')]
                assert (img_prefix == xml_prefix)

                text_polys, text_tags = parse_xml(xml_fn)
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                if text_polys.shape[0] == 0:
                    continue

                # randomly rotate image in four directions
                if FLAGS.rotate_4dir:
                    img, text_polys = random_rotate_4dir(img, text_polys)
                    h, w, _ = img.shape

                # random scale this image
                rd_scale = np.random.choice(random_scale)
                img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale
                # print(rd_scale)
                # random crop a area from image
                if np.random.rand() < background_ratio:
                    # crop background
                    img, text_polys, text_tags = crop_area(img, text_polys, text_tags,
                                                           crop_background=True)
                    if text_polys.shape[0] > 0:
                        # cannot find background
                        continue
                    # pad and resize image
                    new_h, new_w, _ = img.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    img_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    img_padded[:new_h, :new_w, :] = img.copy()
                    img = cv2.resize(img_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 9
                    geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    img, text_polys, text_tags = crop_area(img, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        continue
                    h, w, _ = img.shape

                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = img.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    img_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    img_padded[:new_h, :new_w, :] = img.copy()
                    img = img_padded
                    # resize the image to input size
                    new_h, new_w, _ = img.shape
                    resize_h = input_size
                    resize_w = input_size
                    img = cv2.resize(img, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w / float(new_w)
                    resize_ratio_3_y = resize_h / float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = img.shape
                    score_map, geo_map, training_mask, text_tags = \
                        generate_poly((new_h, new_w), text_polys, text_tags)

                imgs.append(img[:, :, ::-1].astype(np.float32))
                img_fns.append(img_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(imgs) == batch_size:
                    yield imgs, img_fns, score_maps, geo_maps, training_masks
                    imgs = []
                    img_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    """Get batch."""
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    # test data layer
    FLAGS.train_img_list_path = 'test_data_layer/img_list.txt'
    FLAGS.train_xml_list_path = 'test_data_layer/xml_list.txt'
    if not os.path.exists('test_data_layer/visualize'):
        os.makedirs('test_data_layer/visualize')
    data_generator = get_batch(num_workers=4,
                               input_size=FLAGS.input_size,
                               batch_size=1,
                               background_ratio=1.0 / 8)
    for i in range(16):
        data = next(data_generator)
        input_imgs = data[0]
        input_score_maps = data[2]
        input_geo_maps = data[3]
        input_training_masks = data[4]

        # save image
        cv2.imwrite('test_data_layer/visualize/input_img_{}.jpg'.format(i), input_imgs[0][:, :, ::-1])

        # save score map
        cv2.imwrite('test_data_layer/visualize/input_score_map_{}.jpg'.format(i), input_score_maps[0] * 255)

        # save geo map
        for c in range(input_geo_maps[0].shape[-1]):
            input_geo_map_norm = 255 * (input_geo_maps[0][:, :, c] - np.min(input_geo_maps[0][:, :, c])) / \
                                 (np.max(input_geo_maps[0][:, :, c]) - np.min(input_geo_maps[0][:, :, c]) + 1e-5)
            cv2.imwrite('test_data_layer/visualize/input_geo_map_{}_{}.jpg'.format(i, c), input_geo_map_norm)

        # save training masks
        cv2.imwrite('test_data_layer/visualize/input_training_masks_{}.jpg'.format(i), input_training_masks[0] * 255)
