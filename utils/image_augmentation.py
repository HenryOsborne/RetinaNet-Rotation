# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomGrayscale, \
    Compose, ToTensor, ToPILImage

from libs.label_name_dict.label_dict import LabelMap


class ImageAugmentation(object):

    def __init__(self, cfgs):
        self.cfgs = cfgs
        label_map = LabelMap(cfgs)
        self.name2label = label_map.name2label()

    def resize_image(self, image, rw, rh, gtbox_and_label=None):
        img_h, img_w = image.shape[1], image.shape[2]
        image = F.interpolate(image.unsqueeze(0), size=(rw, rh), mode='bilinear')
        if gtbox_and_label is not None:
            x1, y1, x2, y2, x3, y3, x4, y4, label = torch.split(gtbox_and_label, 1, dim=1)
            new_x1 = x1 * rw // img_w
            new_x2 = x2 * rw // img_w
            new_x3 = x3 * rw // img_w
            new_x4 = x4 * rw // img_w

            new_y1 = y1 * rh // img_h
            new_y2 = y2 * rh // img_h
            new_y3 = y3 * rh // img_h
            new_y4 = y4 * rh // img_h
            gtbox_and_label = torch.cat([new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4, label],
                                        dim=1)
            return image.squeeze(0), gtbox_and_label
        else:
            return image.squeeze(0)

    def random_flip_left_right(self, img_tensor, gtboxes_and_label):

        coin = np.random.rand()
        if coin < 0.5:
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            transform = RandomHorizontalFlip(1)
            img_tensor = transform(img_tensor)

            x1, y1, x2, y2, x3, y3, x4, y4, label = torch.split(gtboxes_and_label, 1, dim=1)
            new_x1 = w - x1
            new_x2 = w - x2
            new_x3 = w - x3
            new_x4 = w - x4

            gtboxes_and_label = torch.cat((new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, label), dim=1)

        return img_tensor, gtboxes_and_label

    def random_flip_up_down(self, img_tensor, gtboxes_and_label):

        coin = np.random.rand()
        if coin < 0.5:
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            transform = RandomVerticalFlip(1)
            img_tensor = transform(img_tensor)

            x1, y1, x2, y2, x3, y3, x4, y4, label = torch.split(gtboxes_and_label, 1, dim=1)
            new_y1 = h - y1
            new_y2 = h - y2
            new_y3 = h - y3
            new_y4 = h - y4

            gtboxes_and_label = torch.cat((x1, new_y1, x2, new_y2, x3, new_y3, x4, new_y4, label), dim=1)

        return img_tensor, gtboxes_and_label

    def rotate_img_np(self, img, gtboxes_and_label, r_theta):
        h, w, c = img.shape
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, r_theta, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)  # new W and new H
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rotated_img = cv2.warpAffine(img, M, (nW, nH))

        new_points_list = []
        obj_num = len(gtboxes_and_label)
        for st in range(0, 7, 2):
            points = gtboxes_and_label[:, st:st + 2]
            expand_points = np.concatenate((points, np.ones(shape=(obj_num, 1))), axis=1)
            new_points = np.dot(M, expand_points.T)
            new_points = new_points.T
            new_points_list.append(new_points)
        gtboxes = np.concatenate(new_points_list, axis=1)
        gtboxes_and_label = np.concatenate((gtboxes, gtboxes_and_label[:, -1].reshape(-1, 1)), axis=1)
        gtboxes_and_label = np.asarray(gtboxes_and_label, dtype=np.int32)

        return rotated_img, gtboxes_and_label

    def rotate_img(self, img_tensor, gtboxes_and_label):

        # thetas = tf.constant([-30, -60, -90, 30, 60, 90])
        thetas = tf.range(-90, 90 + 16, delta=15)
        # -90, -75, -60, -45, -30, -15,   0,  15,  30,  45,  60,  75,  90

        theta = tf.random_shuffle(thetas)[0]

        img_tensor, gtboxes_and_label = tf.py_func(self.rotate_img_np,
                                                   inp=[img_tensor, gtboxes_and_label, theta],
                                                   Tout=[tf.float32, tf.int32])

        h, w, c = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1], tf.shape(img_tensor)[2]
        img_tensor = tf.reshape(img_tensor, [h, w, c])
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])

        return img_tensor, gtboxes_and_label

    def random_rotate_img(self, img_tensor, gtboxes_and_label):
        # TODO:未实现旋转图像功能

        return img_tensor, gtboxes_and_label
