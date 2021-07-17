# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from libs.utils import bbox_transform
from libs.utils.iou_rotate import iou_rotate_calculate2


class Loss(object):
    def __init__(self, cfgs, device):
        self.cfgs = cfgs
        self.device = device

    # -------------------------------------- single stage methods---------------------------------------
    def focal_loss(self, labels, pred, anchor_state, alpha=0.25, gamma=2.0):

        # filter out "ignore" anchors
        indices = torch.where(anchor_state != -1)[0]
        labels = labels[indices]
        pred = pred[indices]

        # compute the focal loss
        per_entry_cross_ent = F.binary_cross_entropy_with_logits(input=pred, target=labels, reduction='none')
        prediction_probabilities = torch.sigmoid(pred)
        p_t = ((labels * prediction_probabilities) + (1 - labels) * (1 - prediction_probabilities))
        modulating_factor = 1.0
        if gamma:
            modulating_factor = torch.pow(1.0 - p_t, gamma)
        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = (labels * alpha + (1 - labels) * (1 - alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)

        # compute the normalizer: the number of positive anchors
        normalizer = torch.where(anchor_state == 1)[0].detach()
        normalizer = float(normalizer.shape[0])
        normalizer = max(1.0, normalizer)

        loss = torch.sum(focal_cross_entropy_loss) / normalizer

        return loss

    def smooth_l1_loss(self, targets, preds, anchor_state, sigma=3.0, weight=None):
        sigma_squared = sigma ** 2
        indices = torch.where(anchor_state == 1)[0]
        preds = preds[indices]
        targets = targets[indices]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds - targets
        regression_diff = torch.abs(regression_diff)

        regression_loss = torch.where(
            regression_diff < 1.0 / sigma_squared,
            0.5 * sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # regression_loss = tf.reshape(regression_loss, [-1, 5])
        # lx, ly, lh, lw, ltheta = tf.unstack(regression_loss, axis=-1)
        # regression_loss = tf.transpose(tf.stack([lx*1., ly*1., lh*10., lw*1., ltheta*1.]))

        if weight is not None:
            regression_loss = regression_loss.sum(dim=-1)
            weight = weight[indices]
            regression_loss *= weight

        normalizer = torch.where(anchor_state == 1)[0].detach()
        normalizer = float(normalizer.shape[0])
        normalizer = max(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return regression_loss.sum() / normalizer

    def iou_smooth_l1_loss_log(self, targets, preds, anchor_state, target_boxes, anchors, sigma=3.0, is_refine=False):
        if self.cfgs.METHOD == 'H' and not is_refine:
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * torch.ones_like(x_c)
            anchors = torch.stack([x_c, y_c, w, h, theta], dim=1)

        sigma_squared = sigma ** 2
        indices = torch.where(anchor_state == 1)[0]

        preds = preds[indices]
        targets = targets[indices]
        target_boxes = target_boxes[indices]
        anchors = anchors[indices]

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds - targets
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            regression_diff < 1.0 / sigma_squared,
            0.5 * sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        overlaps = iou_rotate_calculate2(boxes_pred.reshape(-1, 5), target_boxes[:, :-1].reshape(-1, 5))
        overlaps = torch.as_tensor(overlaps, dtype=torch.float32, device=self.device).reshape(-1, 1)

        regression_loss = regression_loss.sum(dim=1).reshape(-1, 1)
        # -ln(x)
        iou_factor = (-1 * torch.log(overlaps)).detach() / (regression_loss + self.cfgs.EPSILON).detach()
        # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

        normalizer = torch.where(anchor_state == 1)[0].detach()
        normalizer = float(normalizer.shape[0])
        normalizer = max(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return (regression_loss * iou_factor).sum() / normalizer

    def iou_smooth_l1_loss_exp(self, targets, preds, anchor_state, target_boxes, anchors, sigma=3.0, alpha=1.0,
                               beta=1.0, is_refine=False):
        if self.cfgs.METHOD == 'H' and not is_refine:
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            h = anchors[:, 2] - anchors[:, 0] + 1
            w = anchors[:, 3] - anchors[:, 1] + 1
            theta = -90 * torch.ones_like(x_c)
            anchors = torch.stack([x_c, y_c, w, h, theta], dim=1)

        sigma_squared = sigma ** 2
        indices = torch.where(anchor_state == 1)[0]

        preds = preds[indices]
        targets = targets[indices]
        target_boxes = target_boxes[indices]
        anchors = anchors[indices]

        boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = preds - targets
        regression_diff = torch.abs(regression_diff)
        regression_loss = torch.where(
            regression_diff < 1.0 / sigma_squared,
            0.5 * sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        overlaps = iou_rotate_calculate2(boxes_pred.reshape(-1, 5), target_boxes[:, :-1].reshape(-1, 5))
        overlaps = torch.as_tensor(overlaps, dtype=torch.float32, device=self.device).reshape(-1, 1)

        regression_loss = regression_loss.sum(dim=1).reshape(-1, 1)
        # 1-exp(1-x)
        iou_factor = (torch.exp(alpha * (1 - overlaps) ** beta) - 1).detach() / (
                regression_loss + self.cfgs.EPSILON).detach()
        # iou_factor = tf.stop_gradient(1-overlaps) / (tf.stop_gradient(regression_loss) + cfgs.EPSILON)
        # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

        normalizer = torch.where(anchor_state == 1)[0].detach()
        normalizer = float(normalizer.shape[0])
        normalizer = max(1.0, normalizer)

        # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
        # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

        return (regression_loss * iou_factor).sum() / normalizer
