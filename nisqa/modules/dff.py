# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import torch.nn as nn
import torch.nn.functional as F


class DFF(nn.Module):
    """
    DFF: Deep Feed-Forward network that was used as baseline framwise model as
    comparision to the CNN.
    """

    def __init__(self,
                 cnn_seg_length,
                 ms_n_mels,
                 dropout,
                 fc_out_h=4096,
                 ):
        super().__init__()
        self.name = "DFF"

        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h
        self.fan_out = fc_out_h

        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length * ms_n_mels

        self.lin1 = nn.Linear(self.fan_in, self.fc_out_h)
        self.lin2 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin3 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin4 = nn.Linear(self.fc_out_h, self.fc_out_h)

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(self.fc_out_h)
        self.bn3 = nn.BatchNorm1d(self.fc_out_h)
        self.bn4 = nn.BatchNorm1d(self.fc_out_h)
        self.bn5 = nn.BatchNorm1d(self.fc_out_h)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(x)
        x = x.view(-1, self.fan_in)

        x = F.relu(self.bn2(self.lin1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.lin2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.lin3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.lin4(x)))

        return x
