# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, cfgs, device):
        super(Sampler, self).__init__()
        self.cfgs = cfgs
        self.device = device
