# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import torch
import torch.nn as nn


class Fusion(torch.nn.Module):
    """
    Fusion: Used by the double-ended NISQA_DE model and used to fuse the
    degraded and reference features.
    """

    def __init__(self, fuse_dim=None, in_feat=None, fuse=None):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.fuse = fuse

        if self.fuse == 'x/y/-':
            self.fan_out = 3 * in_feat
        elif self.fuse == '+/-':
            self.fan_out = 2 * in_feat
        elif self.fuse == 'x/y':
            self.fan_out = 2 * in_feat
        else:
            raise NotImplementedError

        if self.fuse_dim:
            self.lin_fusion = nn.Linear(self.fan_out, self.fuse_dim)
            self.fan_out = fuse_dim

    def forward(self, x, y):

        if self.fuse == 'x/y/-':
            x = torch.cat((x, y, x - y), 2)
        elif self.fuse == '+/-':
            x = torch.cat((x + y, x - y), 2)
        elif self.fuse == 'x/y':
            x = torch.cat((x, y), 2)
        else:
            raise NotImplementedError

        if self.fuse_dim:
            x = self.lin_fusion(x)

        return x
