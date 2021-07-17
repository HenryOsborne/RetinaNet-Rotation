# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


class Sampler(object):
    def __init__(self, cfgs, device):
        self.cfgs = cfgs
        self.device = device
