# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import torch.nn as nn
from .self_attention import SelfAttention
from .lstm import LSTM


# %% Time Dependency
class TimeDependency(nn.Module):
    """
    TimeDependency: The main time-dependency module. It loads either an LSTM
    or self-attention network for time-dependency modelling of the framewise
    features. This module can also be skipped.
    """

    def __init__(self,
                 input_size,
                 td="self_att",
                 sa_d_model=512,
                 sa_nhead=8,
                 sa_pos_enc=None,
                 sa_num_layers=6,
                 sa_h=2048,
                 sa_dropout=0.1,
                 lstm_h=128,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 lstm_bidirectional=True,
                 ):
        super().__init__()

        if td == "self_att":
            self.model = SelfAttention(
                input_size=input_size,
                d_model=sa_d_model,
                nhead=sa_nhead,
                pos_enc=sa_pos_enc,
                num_layers=sa_num_layers,
                sa_h=sa_h,
                dropout=sa_dropout,
                activation="relu"
            )
            self.fan_out = sa_d_model

        elif td == "lstm":
            self.model = LSTM(
                input_size,
                lstm_h=lstm_h,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                bidirectional=lstm_bidirectional,
            )
            self.fan_out = self.model.fan_out

        elif (td is None) or (td == "skip"):
            self.model = self._skip
            self.fan_out = input_size
        else:
            raise NotImplementedError("Time dependency option not available")

    def _skip(self, x, n_wins):
        return x, n_wins

    def forward(self, x, n_wins):
        x, n_wins = self.model(x, n_wins)
        return x, n_wins
