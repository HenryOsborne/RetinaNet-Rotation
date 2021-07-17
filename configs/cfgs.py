# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import os

# from dataloader.pretrained_weights.pretrain_zoo import PretrainModelZoo


# log print
SHOW_TRAIN_INFO_INTE = 20
SMRY_ITER = 200
ADD_BOX_IN_TENSORBOARD = True

# learning policy
EPSILON = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
WARM_EPOCH = 1.0 / 4.0

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

# schedule
BATCH_SIZE = 1
BATCH_SIZE_VAL = 1
GPU_GROUP = "0"
DEVICE = 'cuda:0'  # 'cpu' 'cuda:0'
NUM_GPU = len(GPU_GROUP.strip().split(','))
LR = 1e-3
SAVE_WEIGHTS_INTE = 27000 * 2
DECAY_EPOCH = [18, 24, 30]
MAX_EPOCH = 2
DECAY_STEP = np.array(DECAY_EPOCH, np.int32) * SAVE_WEIGHTS_INTE
MAX_ITERATION = SAVE_WEIGHTS_INTE * MAX_EPOCH
WARM_SETP = int(WARM_EPOCH * SAVE_WEIGHTS_INTE)

# dataset
DATASET_NAME = 'DOTA'
CLASS_NUM = 15
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 600

# data augmentation
RESIZE = True
SIZE = (608, 608)
IMG_ROTATE = False
RGB2GRAY = False
VERTICAL_FLIP = False
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = False

# model
ROOT_PATH = os.path.abspath('./')
SUMMARY_PATH = os.path.join(ROOT_PATH, 'output/summary')

# backbone
NET_NAME = 'resnet50_v1d'
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone

# neck
FPN_MODE = 'fpn'
SHARE_NET = True
USE_P5 = True
FPN_CHANNEL = 256

# bbox head
NUM_SUBNET_CONV = 4
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 5.]
ANCHOR_ANGLES = [-90, -75, -60, -45, -30, -15]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True
METHOD = 'H'
ANGLE_RANGE = 90  # 90 or 180
USE_GN = False
PROBABILITY = 0.01

# loss
CLS_WEIGHT = 1.0
REG_WEIGHT = 1.0

# sample
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

# post-processing
NMS = True
NMS_IOU_THRESHOLD = 0.3
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.05
VIS_SCORE = 0.4

# test and eval
TEST_SAVE_PATH = os.path.join(ROOT_PATH, 'output/test_result')
EVALUATE_R_DIR = os.path.join(ROOT_PATH, 'output/evaluate_result_pickle/')
USE_07_METRIC = True
EVAL_THRESHOLD = 0.5

# backbone


VERSION = 'RetinaNet_DOTA_1x_20210715'

"""
RSDet-8p

This is your result for task 1:

mAP: 0.6727423650267537
ap of each class:
plane:0.8839346472596076,
baseball-diamond:0.7104926703230673,
bridge:0.4330823738329618,
ground-track-field:0.6508970563363848,
small-vehicle:0.6849253155621244,
large-vehicle:0.6102196316871491,
ship:0.7961936701620749,
tennis-court:0.8947516994949227,
basketball-court:0.7455840121634438,
storage-tank:0.7672332259150044,
soccer-ball-field:0.5496680505639349,
roundabout:0.6350320699137543,
harbor:0.5842793618604702,
swimming-pool:0.6505404500942658,
helicopter:0.4943012402321392

The submitted information is :

Description: RetinaNet_DOTA_2x_20201128_162w
Username: SJTU-Det
Institute: SJTU
Emailadress: yangxue-2019-sjtu@sjtu.edu.cn
TeamMembers: yangxue

"""
