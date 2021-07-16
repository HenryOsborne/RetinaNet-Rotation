# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import torch

from libs.models.anchor_heads import generate_h_anchors, generate_r_anchors, generate_h_anchors_tf


class GenerateAnchors(object):

    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.mode = mode

    def anchor_generator(self, base_size, feat_h, feat_w, stride, mode):
        if mode == 'H':
            anchors = generate_h_anchors.generate_anchors_pre(feat_h, feat_w, stride,
                                                              np.array(self.cfgs.ANCHOR_SCALES) * stride,
                                                              self.cfgs.ANCHOR_RATIOS, 4.0)

            anchors = torch.tensor(anchors).reshape(-1, 4)
        else:  # error
            anchors = generate_r_anchors.make_anchors(base_anchor_size=base_size,
                                                      anchor_scales=self.cfgs.ANCHOR_SCALES,
                                                      anchor_ratios=self.cfgs.ANCHOR_RATIOS,
                                                      anchor_angles=self.cfgs.ANCHOR_ANGLES,
                                                      featuremap_height=feat_h,
                                                      featuremap_width=feat_w,
                                                      stride=stride)
        return anchors

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
