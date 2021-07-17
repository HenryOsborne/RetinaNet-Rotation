# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from libs.utils import bbox_transform
from libs.models.losses.losses import Loss
from utils.order_points import re_order


class LossRSDet(Loss):

    def modulated_rotation_5p_loss(self, targets, preds, anchor_state, ratios, sigma=3.0):
        targets = targets.reshape(-1, 5)

        sigma_squared = sigma ** 2
        indices = torch.where(anchor_state == 1)[0]  # .reshape(-1,)
        preds = preds[indices]
        targets = targets[indices]
        ratios = ratios[indices]

        normalizer = torch.where(anchor_state == 1)[0].detach()
        normalizer = float(normalizer.shape[0])
        normalizer = max(1.0, normalizer)

        regression_diff = preds - targets
        regression_diff = torch.abs(regression_diff)
        loss1 = torch.where(regression_diff < 1.0 / sigma_squared, 0.5 * sigma_squared * torch.pow(regression_diff, 2),
                            regression_diff - 0.5 / sigma_squared)
        loss1 = loss1.sum(dim=1)

        loss2_1 = preds[:, 0] - targets[:, 0]
        loss2_2 = preds[:, 1] - targets[:, 1]
        # loss2_3 = preds[:, 2] - targets[:, 3] - tf.log(ratios)
        # loss2_4 = preds[:, 3] - targets[:, 2] + tf.log(ratios)
        loss2_3 = preds[:, 2] - targets[:, 3] + torch.log(ratios)
        loss2_4 = preds[:, 3] - targets[:, 2] - torch.log(ratios)
        loss2_5 = torch.min((preds[:, 4] - targets[:, 4] + 1.570796), (targets[:, 4] - preds[:, 4] + 1.570796))

        box_diff_2 = torch.stack([loss2_1, loss2_2, loss2_3, loss2_4, loss2_5], 1)
        abs_box_diff_2 = torch.abs(box_diff_2)
        loss2 = torch.where(abs_box_diff_2 < 1.0 / sigma_squared, 0.5 * sigma_squared * torch.pow(abs_box_diff_2, 2),
                            abs_box_diff_2 - 0.5 / sigma_squared)

        loss2 = loss2.sum(dim=1)
        loss = torch.min(loss1, loss2)
        loss = loss.sum() / normalizer

        return loss

    def modulated_rotation_8p_loss(self, targets, preds, anchor_state, anchors, sigma=3.0):
        targets = targets[:, :-1].reshape(-1, 8)

        sigma_squared = sigma ** 2
        indices = torch.where(anchor_state == 1)[0]  # .reshape(-1,)
        preds = preds[indices]
        targets = targets[indices]
        anchors = anchors[indices]

        if self.cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            w = anchors[:, 2] - anchors[:, 0] + 1
            h = anchors[:, 3] - anchors[:, 1] + 1
            # theta = -90 * tf.ones_like(x_c)
            anchors = torch.stack([x_c, y_c, w, h], dim=1)

        preds = bbox_transform.qbbox_transform_inv(boxes=anchors, deltas=preds)

        targets = re_order(targets.cpu())
        targets = torch.as_tensor(targets, device=self.device).reshape(-1, 8)

        # prepare for normalization
        normalizer = torch.where(anchor_state == 1)[0].detach()
        normalizer = float(normalizer.shape[0])
        normalizer = max(1.0, normalizer)

        # loss1
        loss1_1 = (preds[:, 0] - targets[:, 0]) / anchors[:, 2]
        loss1_2 = (preds[:, 1] - targets[:, 1]) / anchors[:, 3]
        loss1_3 = (preds[:, 2] - targets[:, 2]) / anchors[:, 2]
        loss1_4 = (preds[:, 3] - targets[:, 3]) / anchors[:, 3]
        loss1_5 = (preds[:, 4] - targets[:, 4]) / anchors[:, 2]
        loss1_6 = (preds[:, 5] - targets[:, 5]) / anchors[:, 3]
        loss1_7 = (preds[:, 6] - targets[:, 6]) / anchors[:, 2]
        loss1_8 = (preds[:, 7] - targets[:, 7]) / anchors[:, 3]
        box_diff_1 = torch.stack([loss1_1, loss1_2, loss1_3, loss1_4, loss1_5, loss1_6, loss1_7, loss1_8], dim=1)
        box_diff_1 = torch.abs(box_diff_1)
        loss_1 = torch.where(box_diff_1 < (1.0 / sigma_squared), 0.5 * sigma_squared * torch.pow(box_diff_1, 2),
                             box_diff_1 - 0.5 / sigma_squared)
        loss_1 = loss_1.sum(dim=1)

        # loss2
        loss2_1 = (preds[:, 0] - targets[:, 2]) / anchors[:, 2]
        loss2_2 = (preds[:, 1] - targets[:, 3]) / anchors[:, 3]
        loss2_3 = (preds[:, 2] - targets[:, 4]) / anchors[:, 2]
        loss2_4 = (preds[:, 3] - targets[:, 5]) / anchors[:, 3]
        loss2_5 = (preds[:, 4] - targets[:, 6]) / anchors[:, 2]
        loss2_6 = (preds[:, 5] - targets[:, 7]) / anchors[:, 3]
        loss2_7 = (preds[:, 6] - targets[:, 0]) / anchors[:, 2]
        loss2_8 = (preds[:, 7] - targets[:, 1]) / anchors[:, 3]
        box_diff_2 = torch.stack([loss2_1, loss2_2, loss2_3, loss2_4, loss2_5, loss2_6, loss2_7, loss2_8], 1)
        box_diff_2 = torch.abs(box_diff_2)
        loss_2 = torch.where(box_diff_2 < 1.0 / sigma_squared, 0.5 * sigma_squared * torch.pow(box_diff_2, 2),
                             box_diff_2 - 0.5 / sigma_squared)
        loss_2 = loss_2.sum(dim=1)

        # loss3
        loss3_1 = (preds[:, 0] - targets[:, 6]) / anchors[:, 2]
        loss3_2 = (preds[:, 1] - targets[:, 7]) / anchors[:, 3]
        loss3_3 = (preds[:, 2] - targets[:, 0]) / anchors[:, 2]
        loss3_4 = (preds[:, 3] - targets[:, 1]) / anchors[:, 3]
        loss3_5 = (preds[:, 4] - targets[:, 2]) / anchors[:, 2]
        loss3_6 = (preds[:, 5] - targets[:, 3]) / anchors[:, 3]
        loss3_7 = (preds[:, 6] - targets[:, 4]) / anchors[:, 2]
        loss3_8 = (preds[:, 7] - targets[:, 5]) / anchors[:, 3]
        box_diff_3 = torch.stack([loss3_1, loss3_2, loss3_3, loss3_4, loss3_5, loss3_6, loss3_7, loss3_8], dim=1)
        box_diff_3 = torch.abs(box_diff_3)
        loss_3 = torch.where(box_diff_3 < 1.0 / sigma_squared, 0.5 * sigma_squared * torch.pow(box_diff_3, 2),
                             box_diff_3 - 0.5 / sigma_squared)
        loss_3 = loss_3.sum(dim=1)

        loss = torch.min(torch.min(loss_1, loss_2), loss_3)
        loss = torch.sum(loss) / normalizer

        return loss
