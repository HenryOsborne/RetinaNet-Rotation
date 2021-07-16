# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import cv2
import torch


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles,
                 featuremap_height, featuremap_width, stride, name='make_ratate_anchors'):
    """
    :param base_anchor_size:
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    """
    base_anchor = torch.tensor([0, 0, base_anchor_size, base_anchor_size],
                               dtype=torch.float32)  # [y_center, x_center, h, w]
    ws, hs, angles = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales), anchor_ratios, anchor_angles)
    # per locations ws and hs and thetas

    x_centers = torch.range(0, featuremap_width, dtype=torch.float32) * stride + stride // 2
    y_centers = torch.range(0, featuremap_height, dtype=torch.float32) * stride + stride // 2

    x_centers, y_centers = torch.meshgrid(x_centers, y_centers)

    angles, _ = torch.meshgrid(angles, x_centers)
    ws, x_centers = torch.meshgrid(ws, x_centers)
    hs, y_centers = torch.meshgrid(hs, y_centers)

    anchor_centers = torch.stack([x_centers, y_centers], dim=2)
    anchor_centers = anchor_centers.reshape(-1, 2)

    box_parameters = torch.stack([ws, hs, angles], dim=2)
    box_parameters = box_parameters.reshape(-1, 3)
    anchors = torch.cat([anchor_centers, box_parameters], dim=1)

    return anchors


def enum_scales(base_anchor, anchor_scales):
    anchor_scales = base_anchor * torch.tensor(anchor_scales, dtype=torch.float32).reshape(len(anchor_scales), 1)
    return anchor_scales
    # anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))


def enum_ratios_and_thetas(anchors, anchor_ratios, anchor_angles):
    """
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    """
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    anchor_angles = torch.tensor(anchor_angles, dtype=torch.float32)
    sqrt_ratios = torch.sqrt(torch.tensor(anchor_ratios))

    ws = (ws / sqrt_ratios[:, None]).reshape(-1)
    hs = (hs * sqrt_ratios[:, None]).reshape(-1)

    ws, _ = torch.meshgrid(ws, anchor_angles)
    hs, anchor_angles = torch.meshgrid(hs, anchor_angles)

    anchor_angles = anchor_angles.reshape(-1, 1)

    ws = ws.reshape(-1, 1)
    hs = hs.reshape(-1, 1)

    return ws, hs, anchor_angles


if __name__ == '__main__':
    import os
    from libs.configs import cfgs
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from libs.utils.show_box_in_tensor import DrawBoxTensor

    drawer = DrawBoxTensor(cfgs)

    base_anchor_size = 256
    anchor_scales = [1.]
    anchor_ratios = [0.5, 2.0, 1 / 3, 3, 1 / 5, 5, 1 / 8, 8]
    anchor_angles = [-90, -75, -60, -45, -30, -15]
    base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
    tmp1 = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales), anchor_ratios, anchor_angles)
    anchors = make_anchors(32,
                           [2.], [2.0, 1 / 2], anchor_angles,
                           featuremap_height=800 // 8,
                           featuremap_width=800 // 8,
                           stride=8)

    # anchors = make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
    #                        anchor_scales=cfgs.ANCHOR_SCALES,
    #                        anchor_ratios=cfgs.ANCHOR_RATIOS,
    #                        anchor_angles=cfgs.ANCHOR_ANGLES,
    #                        featuremap_height=800 // 16,
    #                        featuremap_width=800 // 16,
    #                        stride=cfgs.ANCHOR_STRIDE[0],
    #                        name="make_anchors_forRPN")

    img = tf.zeros([800, 800, 3])
    img = tf.expand_dims(img, axis=0)

    img1 = drawer.only_draw_boxes(img, anchors[9100:9110], 'r')

    with tf.Session() as sess:
        temp1, _img1 = sess.run([anchors, img1])

        _img1 = _img1[0]

        cv2.imwrite('rotate_anchors.jpg', _img1)
        cv2.waitKey(0)

        print(temp1)
        print('debug')
