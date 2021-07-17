from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from libs.models.samplers.samper import Sampler
from libs.utils.cython_utils.cython_bbox import bbox_overlaps
from libs.utils.rbbox_overlaps import rbbx_overlaps
from libs.utils import bbox_transform
from utils.order_points import sort_corners


class AnchorSamplerRSDet(Sampler):

    def anchor_target_layer(self, gt_boxes_h, gt_boxes_r, anchors, gpu_id=0):

        anchor_states = torch.zeros((anchors.shape[0],), device=self.device)
        labels = torch.zeros((anchors.shape[0], self.cfgs.CLASS_NUM), device=self.device)
        if gt_boxes_r.shape[0]:
            # [N, M]

            if self.cfgs.METHOD == 'H':
                overlaps = bbox_overlaps(np.ascontiguousarray(anchors.cpu(), dtype=np.float),
                                         np.ascontiguousarray(gt_boxes_h.cpu(), dtype=np.float))
            else:
                overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                         np.ascontiguousarray(gt_boxes_r[:, :-1], dtype=np.float32), gpu_id)

            overlaps = torch.as_tensor(overlaps, device=self.device)
            argmax_overlaps_inds = torch.argmax(overlaps, dim=1)
            max_overlaps = overlaps[torch.arange(overlaps.shape[0]), argmax_overlaps_inds]

            # compute box regression targets
            target_boxes = gt_boxes_r[argmax_overlaps_inds]

            positive_indices = max_overlaps >= self.cfgs.IOU_POSITIVE_THRESHOLD
            ignore_indices = (max_overlaps > self.cfgs.IOU_NEGATIVE_THRESHOLD) & ~positive_indices

            anchor_states[ignore_indices] = -1
            anchor_states[positive_indices] = 1

            # compute target class labels
            idx = target_boxes[positive_indices, -1].type(torch.int64)
            labels[positive_indices, idx - 1] = 1
        else:
            # no annotations? then everything is background
            target_boxes = torch.zeros((anchors.shape[0], gt_boxes_r.shape[1]))

        # if self.cfgs.METHOD == 'H':
        #     w = anchors[:, 2] - anchors[:, 0] + 1
        #     h = anchors[:, 3] - anchors[:, 1] + 1
        #     x1 = anchors[:, 0]
        #     y1 = anchors[:, 1]
        #     x2 = anchors[:, 2]
        #     y2 = anchors[:, 1]
        #     x3 = anchors[:, 2]
        #     y3 = anchors[:, 3]
        #     x4 = anchors[:, 0]
        #     y4 = anchors[:, 3]
        #     anchors = np.stack([x1, y1, x2, y2, x3, y3, x4, y4, w, h]).transpose()
        #
        # target_delta = bbox_transform.qbbox_transform(ex_rois=anchors, gt_rois=target_boxes)
        # labels = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        # anchor_states = torch.as_tensor(anchor_states, dtype=torch.float32, device=self.device)

        return labels, anchor_states, target_boxes
