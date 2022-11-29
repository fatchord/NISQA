# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import os
import time
import multiprocessing
import copy
import math

import librosa as lb
import numpy as np
import pandas as pd;

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SkipCNN(nn.Module):
    '''
    SkipCNN: Can be used to skip the framewise modelling stage and directly
    apply an LSTM or Self-Attention network.
    '''

    def __init__(
            self,
            cnn_seg_length,
            ms_n_mels,
            fc_out_h
    ):
        super().__init__()

        self.name = 'SkipCNN'
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length * ms_n_mels
        self.bn = nn.BatchNorm2d(1)

        if fc_out_h is not None:
            self.linear = nn.Linear(self.fan_in, fc_out_h)
            self.fan_out = fc_out_h
        else:
            self.linear = nn.Identity()
            self.fan_out = self.fan_in

    def forward(self, x):
        x = self.bn(x)
        x = x.view(-1, self.fan_in)
        x = self.linear(x)
        return x

