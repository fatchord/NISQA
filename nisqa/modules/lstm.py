# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    """
    LSTM: The main LSTM module that can be used as a time-dependency model.
    """

    def __init__(self,
                 input_size,
                 lstm_h=128,
                 num_layers=1,
                 dropout=0.1,
                 bidirectional=True
                 ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_h,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.fan_out = num_directions * lstm_h

    def forward(self, x, n_wins):

        x = pack_padded_sequence(
            x,
            n_wins.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]

        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=0.0,
            total_length=n_wins.max())

        return x, n_wins
