# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from .standard_cnn import StandardCNN
from .adapt_cnn import AdaptCNN
from .skip_cnn import SkipCNN
from .dff import DFF


# %% Framewise
class Framewise(nn.Module):
    """
    Framewise: The main framewise module. It loads either a CNN or feed-forward
    network for framewise modelling of the Mel-spec segments. This module can
    also be skipped by loading the SkipCNN module. There are two CNN modules
    available. AdaptCNN with adaptive maxpooling and the StandardCNN module.
    However, they could also be replaced with new modules, such as PyTorch
    implementations of ResNet or Alexnet.
    """

    def __init__(
            self,
            cnn_model,
            ms_seg_length=15,
            ms_n_mels=48,
            c_out_1=16,
            c_out_2=32,
            c_out_3=64,
            kernel_size=3,
            dropout=0.2,
            pool_1=[24, 7],
            pool_2=[12, 5],
            pool_3=[6, 3],
            fc_out_h=None,
    ):
        super().__init__()

        if cnn_model == "adapt":
            self.model = AdaptCNN(
                input_channels=1,
                c_out_1=c_out_1,
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size,
                dropout=dropout,
                pool_1=pool_1,
                pool_2=pool_2,
                pool_3=pool_3,
                fc_out_h=fc_out_h,
            )
        elif cnn_model == "standard":
            assert ms_n_mels == 48, "ms_n_mels is {} and should be 48, use adaptive model or change ms_n_mels".format(
                ms_n_mels)
            assert ms_seg_length == 15, "ms_seg_len is {} should be 15, use adaptive model or change ms_seg_len".format(
                ms_seg_length)
            assert ((kernel_size == 3) or (kernel_size == (
            3, 3))), "cnn_kernel_size is {} should be 3, use adaptive model or change cnn_kernel_size".format(
                kernel_size)
            self.model = StandardCNN(
                input_channels=1,
                c_out_1=c_out_1,
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size,
                dropout=dropout,
                fc_out_h=fc_out_h,
            )
        elif cnn_model == "dff":
            self.model = DFF(ms_seg_length, ms_n_mels, dropout, fc_out_h)
        elif (cnn_model is None) or (cnn_model == "skip"):
            self.model = SkipCNN(ms_seg_length, ms_n_mels, fc_out_h)
        else:
            raise NotImplementedError("Framwise model not available")

    def forward(self, x, n_wins):
        (bs, length, channels, height, width) = x.shape
        x_packed = pack_padded_sequence(
            x,
            n_wins.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        x = self.model(x_packed.data)
        x = x_packed._replace(data=x)
        x, _ = pad_packed_sequence(
            x,
            batch_first=True,
            padding_value=0.0,
            total_length=n_wins.max())
        return x

