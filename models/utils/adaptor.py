# -*- coding: utf-8 -*-
import torch.nn as nn
from models.pose.hourglass.base.layers import Conv


class MultiAdaptor(object):
    def __init__(self, device, stream_num, feature_dim):
        self.unit = 10
        self.stream_num = stream_num
        self.adaptors = nn.ModuleList([
            nn.Sequential(
                Conv(feature_dim, feature_dim, 1, bn=False, relu=True),
                Conv(feature_dim, feature_dim, 1, bn=False, relu=True),
                Conv(feature_dim, feature_dim, 1, bn=False, relu=True)
            ) for stIdx in range(self.stream_num)]).to(device)

    def forward(self, idx, x):
        return self.adaptors[idx](x)
