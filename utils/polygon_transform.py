# _*_ coding: utf-8 _*_
"""Polygon transform"""
import numpy as np


def restore_poly(origin, geo_maps):
    """
    Restore polygon.
    """
    assert geo_maps.shape[-1] % 2 == 0
    point_num = int(geo_maps.shape[-1] / 2)

    origin_tile = np.tile(origin, (1, point_num)) # (n, point_num * 2)
    pred_polys = origin_tile - geo_maps
    pred_polys = pred_polys.reshape((-1, point_num, 2)) # (n, point_num, 2)

    return pred_polys


def poly_transform_inv(score_maps, geo_maps):
    """
    Polygon transform inverse.
    """
    assert geo_maps.shape[-1] % 2 == 0
    point_num = int(geo_maps.shape[-1] / 2)

    if score_maps.shape[0] == 0:
        return np.zeros((0, point_num * 2), dtype=geo_maps.dtype)

    batch_inds, y_inds, x_inds, _ = np.where(score_maps > 0)
    dxs, dys = [], []
    for i in range(point_num):
        dxs.append(geo_maps[batch_inds, y_inds, x_inds, 2 * i])
        dys.append(geo_maps[batch_inds, y_inds, x_inds, 2 * i + 1])

    # restore x_inds and y_inds to input size
    input_x_inds = 4 * x_inds
    input_y_inds = 4 * y_inds

    # polygon
    xs, ys = [], []
    for i in range(point_num):
        xs.append(input_x_inds - dxs[i])
        ys.append(input_y_inds - dys[i])

    # TODO
    pred_polys = np.vstack((xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], xs[3], ys[3])).transpose()
    scores = score_maps[batch_inds, y_inds, x_inds, 0].reshape((-1, 1))
    batch_inds = batch_inds.reshape((-1, 1))

    return pred_polys, scores, batch_inds


def clip_polys(polys, img_shape):
    """
    Clip polygons to image boundaries.

    x >= 0 and x <= img_shape[1] - 1
    y >= 0 and y <= img_shape[0] - 1
    """

    assert polys.shape[-1] % 2 == 0
    point_num = int(polys.shape[-1] / 2)

    for i in range(point_num):
        polys[:, i * 2]     = np.maximum(np.minimum(polys[:, i * 2],     img_shape[1] - 1), 0)
        polys[:, i * 2 + 1] = np.maximum(np.minimum(polys[:, i * 2 + 1], img_shape[0] - 1), 0)

    return polys


def filter_polys(polys, min_size):
    """
    Remove all polys with any side smaller than min_size.
    """

    assert polys.shape[-1] % 2 == 0
    point_num = int(polys.shape[-1] / 2)

    lens = []
    for i in range(point_num):
        x_ind1 = 2 * i
        x_ind2 = (2 * i + 2) % (2 * point_num)
        lens.append(np.linalg.norm(polys[:, x_ind1:(x_ind1 + 1)] -\
                                   polys[:, x_ind2:(x_ind2 + 1)], axis=1))

    # TODO
    keep = np.where((lens[0] > min_size) & \
                    (lens[1] > min_size) & \
                    (lens[2] > min_size) & \
                    (lens[3] > min_size))[0]

    return keep