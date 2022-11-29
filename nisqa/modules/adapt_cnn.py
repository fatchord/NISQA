# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import torch.nn as nn
import torch.nn.functional as F


class AdaptCNN(nn.Module):
    """
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    """

    def __init__(self,
                 input_channels,
                 c_out_1,
                 c_out_2,
                 c_out_3,
                 kernel_size,
                 dropout,
                 pool_1,
                 pool_2,
                 pool_3,
                 fc_out_h=20,
                 ):
        super().__init__()
        self.name = "CNN_adapt"

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        # Set kernel width of last conv layer to last pool width to
        # downsample width to one.
        self.kernel_size_last = (self.kernel_size[0], self.pool_3[1])

        # kernel_size[1]=1 can be used for seg_length=1 -> corresponds to
        # 1D conv layer, no width padding needed.
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1, 0)
        else:
            self.cnn_pad = (1, 1)

        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.c_out_1,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(
            self.conv1.out_channels,
            self.c_out_2,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(
            self.conv2.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(
            self.conv3.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = nn.Conv2d(
            self.conv4.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=self.cnn_pad)

        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)

        self.conv6 = nn.Conv2d(
            self.conv5.out_channels,
            self.c_out_3,
            self.kernel_size_last,
            padding=(1, 0))

        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))

        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))

        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])

        if self.fc_out_h:
            x = self.fc(x)
        return x
