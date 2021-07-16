import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from retinanet.utils import BasicBlock, Bottleneck
from libs.models.anchor_heads.generate_anchors import GenerateAnchors
from libs.models.samplers.rsdet.anchor_sampler_rsdet_8p import AnchorSamplerRSDet
from libs.models.losses.losses_rsdet import LossRSDet
from libs.utils import bbox_transform, nms_rotate
from libs.utils.coordinate_convert import backward_convert

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5_x = self.P5_1(c5)
        p5_upsampled_x = self.P5_upsampled(p5_x)
        p5_x = self.P5_2(p5_x)

        p4_x = self.P4_1(c4)
        p4_x = p5_upsampled_x + p4_x
        p4_upsampled_x = self.P4_upsampled(p4_x)
        p4_x = self.P4_2(p4_x)

        p3_x = self.P3_1(c3)
        p3_x = p3_x + p4_upsampled_x
        p3_x = self.P3_2(p3_x)

        p6_x = self.P6(c5)

        p7_x = self.P7_1(p6_x)
        p7_x = self.P7_2(p7_x)

        return [p3_x, p4_x, p5_x, p6_x, p7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, cfgs, feature_size=256):
        super(RegressionModel, self).__init__()

        if cfgs.METHOD == 'H':
            num_anchors = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            num_anchors = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 8, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 5*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 8)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, cfgs, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = cfgs.CLASS_NUM
        if cfgs.METHOD == 'H':
            self.num_anchors = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        else:
            self.num_anchors = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS) * len(cfgs.ANCHOR_ANGLES)

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, self.num_anchors * self.num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out_scores = self.output(out)
        out_probs = self.output_act(out_scores)

        # out is B x C x W x H, with C = n_classes * n_anchors
        out1 = out_scores.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out1 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out1 = out1.contiguous().view(x.shape[0], -1, self.num_classes)
        # out1 = out1.contiguous().view(-1, self.num_classes)

        out2 = out_probs.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out2.shape
        out2 = out2.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out2 = out2.contiguous().view(x.shape[0], -1, self.num_classes)
        # out2 = out2.contiguous().view(-1, self.num_classes)

        return out1, out2


class ResNet(nn.Module):

    def __init__(self, cfgs, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.cfgs = cfgs

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256, cfgs)

        self.classificationModel = ClassificationModel(256, cfgs)

        self.anchors = GenerateAnchors(cfgs, cfgs.METHOD)
        self.anchor_sampler_rsdet = AnchorSamplerRSDet(cfgs)

        self.losses = LossRSDet(cfgs)
        self.losses_dict = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, gtboxes_batch_r, gtboxes_batch_h):

        if self.training:
            img_batch = inputs
            gtboxes_batch_h = gtboxes_batch_h  # list to tensor
            gtboxes_batch_r = gtboxes_batch_r
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        regression = regression.squeeze(dim=0)

        classification = [self.classificationModel(feature) for feature in features]
        cls_scores = torch.cat([cls[0] for cls in classification], dim=1)
        cls_probs = torch.cat([cls[1] for cls in classification], dim=1)
        cls_scores = cls_scores.squeeze(dim=0)
        cls_probs = cls_probs.squeeze(dim=0)

        anchors = self.anchors.generate_all_anchor(features)
        anchors = torch.cat(anchors, dim=0)

        if self.training:
            labels, anchor_states, target_boxes = self.anchor_sampler_rsdet.anchor_target_layer(gtboxes_batch_h,
                                                                                                gtboxes_batch_r,
                                                                                                anchors, gpu_id=0)
            labels, anchor_states, target_boxes = \
                torch.from_numpy(labels), torch.from_numpy(anchor_states), torch.from_numpy(target_boxes)

            cls_loss = self.losses.focal_loss(labels, cls_scores, anchor_states)
            # labels[num_anchors,num_class] one-hot encoding
            reg_loss = self.losses.modulated_rotation_8p_loss(target_boxes, regression, anchor_states, anchors)

            self.losses_dict['cls_loss'] = cls_loss * self.cfgs.CLS_WEIGHT
            self.losses_dict['reg_loss'] = reg_loss * self.cfgs.REG_WEIGHT

            return self.losses_dict

        else:

            boxes, scores, category = self.postprocess_detctions(rpn_bbox_pred=regression,
                                                                 rpn_cls_prob=cls_probs,
                                                                 anchors=anchors,
                                                                 gpu_id=0)
            boxes = boxes.detach()
            scores = scores.detach()
            category = category.detach()

            return boxes, scores, category

        # if self.training:
        #     return boxes, scores, category, self.losses_dict
        # else:
        #     return boxes, scores, category

    def postprocess_detctions(self, rpn_bbox_pred, rpn_cls_prob, anchors, gpu_id):

        def filter_detections(boxes, scores, is_training, gpu_id):

            if is_training:
                indices = torch.where(scores > self.cfgs.VIS_SCORE)[0]
            else:
                indices = torch.where(scores > self.cfgs.FILTERED_SCORE)[0]

            if self.cfgs.NMS:
                filtered_boxes = boxes[indices]
                filtered_scores = scores[indices]

                filtered_boxes = backward_convert(filtered_boxes, False)

                # [x, y, w, h, theta]
                max_output_size = 4000 if 'DOTA' in self.cfgs.NET_NAME else 200
                nms_indices = nms_rotate.nms_rotate(decode_boxes=filtered_boxes,
                                                    scores=filtered_scores.reshape(-1, ),
                                                    iou_threshold=self.cfgs.NMS_IOU_THRESHOLD,
                                                    max_output_size=100 if is_training else max_output_size,
                                                    use_gpu=not self.training,
                                                    gpu_id=gpu_id)

                # filter indices based on NMS
                indices = indices[nms_indices]

            # add indices to list of all indices
            # return indices
            return indices

        if self.cfgs.METHOD == 'H':
            x_c = (anchors[:, 2] + anchors[:, 0]) / 2
            y_c = (anchors[:, 3] + anchors[:, 1]) / 2
            w = anchors[:, 2] - anchors[:, 0] + 1
            h = anchors[:, 3] - anchors[:, 1] + 1
            anchors = torch.stack([x_c, y_c, w, h], dim=1)

        boxes_pred = bbox_transform.qbbox_transform_inv(boxes=anchors, deltas=rpn_bbox_pred)

        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, self.cfgs.CLASS_NUM):
            indices = filter_detections(boxes_pred, rpn_cls_prob[:, j], self.training, gpu_id)
            tmp_boxes_pred = boxes_pred[indices].reshape(-1, 8)
            tmp_scores = rpn_cls_prob[:, j][indices].reshape(-1, )

            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(torch.ones_like(tmp_scores) * (j + 1))

        return_boxes_pred = torch.cat(return_boxes_pred, dim=0)
        return_scores = torch.cat(return_scores, dim=0)
        return_labels = torch.cat(return_labels, dim=0)

        return return_boxes_pred, return_scores, return_labels


def resnet18(cfgs, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(cfgs, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(cfgs, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(cfgs, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(cfgs, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(cfgs, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(cfgs, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(cfgs, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(cfgs, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(cfgs, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
