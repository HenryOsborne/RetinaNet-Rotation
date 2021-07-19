# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import torch

from libs.models.anchor_heads import generate_h_anchors, generate_r_anchors
import cfgs


class GenerateAnchors(object):

    def __init__(self, cfgs, mode, device):
        self.cfgs = cfgs
        self.mode = mode
        self.device = device

    def anchor_generator(self, base_size, feat_h, feat_w, stride, mode):
        if mode == 'H':
            anchors = generate_h_anchors.generate_anchors_pre(feat_h, feat_w, stride,
                                                              np.array(self.cfgs.ANCHOR_SCALES) * stride,
                                                              self.cfgs.ANCHOR_RATIOS, 4.0)

            anchors = torch.as_tensor(anchors).reshape(-1, 4)
        else:  # error
            anchors = generate_r_anchors.make_anchors(base_anchor_size=base_size,
                                                      anchor_scales=self.cfgs.ANCHOR_SCALES,
                                                      anchor_ratios=self.cfgs.ANCHOR_RATIOS,
                                                      anchor_angles=self.cfgs.ANCHOR_ANGLES,
                                                      featuremap_height=feat_h,
                                                      featuremap_width=feat_w,
                                                      stride=stride)
        return anchors.to(self.device)

    def generate_all_anchor(self, feature_pyramid):

        '''
            (level, base_anchor_size) tuple:
            (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
        '''

        anchor_list = []
        for i, (base_size, stride) in enumerate(zip(self.cfgs.BASE_ANCHOR_SIZE_LIST, self.cfgs.ANCHOR_STRIDE)):
            feat_h, feat_w = feature_pyramid[i].shape[2], feature_pyramid[i].shape[3]
            anchor_tmp = self.anchor_generator(base_size, feat_h, feat_w, stride, self.mode)
            anchor_list.append(anchor_tmp)

        return anchor_list

    def generate_all_anchor_test(self, feature_pyramid):

        '''
            (level, base_anchor_size) tuple:
            (P3, 32), (P4, 64), (P5, 128), (P6, 256), (P7, 512)
        '''

        anchor_list = []
        for i, (base_size, stride) in enumerate(zip(self.cfgs.BASE_ANCHOR_SIZE_LIST, self.cfgs.ANCHOR_STRIDE)):
            feat_h, feat_w = feature_pyramid[i], feature_pyramid[i]
            anchor_tmp = self.anchor_generator(base_size, feat_h, feat_w, stride, self.mode)
            anchor_list.append(anchor_tmp)

        return anchor_list


if __name__ == '__main__':
    gen = GenerateAnchors(cfgs, 'H', torch.device('cpu'))
    feat = [76, 38, 19, 10, 5]
    anchor = gen.generate_all_anchor_test(feat)
    anchor = [i.numpy() for i in anchor]
    print(anchor)
