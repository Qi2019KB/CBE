# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import random
from models.pose.hourglass.base.layers import Conv
from models.utils.adaptor import MultiAdaptor


class MultiStream(object):
    def __init__(self, device, stream_num, feature_shape, split_factor=0.1, is_expand=False, noisy_factor=0.2):
        self.is_expand = is_expand
        self.noisy_factor = noisy_factor
        self.stream_num = stream_num
        self.feature_shape = feature_shape
        self.expand_shape, mask_cNum = self._cNum(split_factor)
        self.switch = self._switch(mask_cNum)
        self.mask = self._mask()
        if is_expand:
            self.pre = Conv(self.feature_shape[0], self.expand_shape[0], 1, bn=False, relu=False).to(device)
        else:
            self.pre = Conv(self.feature_shape[0], self.feature_shape[0], 1, bn=False, relu=False).to(device)

    def _cNum(self, split_factor):
        cNum = self.feature_shape[0]
        mask_cNum = int(cNum * split_factor)
        if self.is_expand:
            expand_cNum = mask_cNum * (self.stream_num - 1)
            expand_shape = [self.feature_shape[0] + expand_cNum, self.feature_shape[1], self.feature_shape[2]]
        else:
            expand_shape = self.feature_shape
        return expand_shape, mask_cNum

    def _switch(self, mask_cNum):
        cNum = self.expand_shape[0] if self.is_expand else self.feature_shape[0]
        switchs_bg = [item for item in range(0, mask_cNum * self.stream_num)]
        switch, switchs_fg = [[] for stIdx in range(self.stream_num)], []
        commIdx = -1  # idx of item in switchs_comm
        for cIdx in range(0, cNum):
            if cIdx in switchs_bg:
                commIdx += 1
                switch[commIdx // mask_cNum].append(cIdx)
            else:
                switchs_fg.append(cIdx)
        return switch + [switchs_bg, switchs_fg]

    def _mask(self):
        # set channel_switchs
        ms_masks = []
        for stIdx in range(self.stream_num):
            mask = torch.ones(self.expand_shape) if self.is_expand else torch.ones(self.feature_shape)
            mask[self.switch[-2]] = 0
            mask[self.switch[stIdx]] = 1
            ms_masks.append(mask)
        return ms_masks

    def forward(self, idx, x, active_param=None):
        bs, c, _, _ = x.size()
        if self.noisy_factor > 0:
            x = self._noisy_mean(x) if active_param is None else self._noisy_mean(x, active_param)
        if self.is_expand:
            ms_f = x[:, self.switch[idx] + self.switch[-1]]
        else:
            # sample in channel dimension by mask.
            ms_mask = self.mask[idx].unsqueeze(0)
            ms_mask = ms_mask.repeat(bs, 1, 1, 1)
            ms_f = x * ms_mask.to(x.device, non_blocking=True)

        # get the private channel index.
        switchs_fg = self.switch[idx]
        ms_f_p = ms_f[:, switchs_fg]
        return ms_f, ms_f_p

    def _noisy_mean(cls, input, active_param=None, prob=0.5):
        if random.random() <= prob:
            mu = input.mean()
            nf = cls.noisy_factor if active_param is None else cls.noisy_factor*(1+active_param)
            input = random.uniform(1-nf, 1+nf) * (input - mu) + mu
            input.add_(random.uniform(-nf, nf)).clamp_(0, 1)
        return input
