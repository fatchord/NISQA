# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""

import torch.nn as nn
import torch.nn.functional as F


class StandardCNN(nn.Module):
    """
    StandardCNN: CNN with fixed maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module requires a fixed
    input dimension of 48x15.
    """

    def __init__(
            self,
            input_channels,
            c_out_1,
            c_out_2,
            c_out_3,
            kernel_size,
            dropout,
            fc_out_h=None
    ):
        super().__init__()

        self.name = "CNN_standard"

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_size = 2
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.output_width = 2  # input width 15 pooled 3 times
        self.output_height = 6  # input height 48 pooled 3 times

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        self.pool_first = nn.MaxPool2d(
            self.pool_size,
            stride=self.pool_size,
            padding=(0, 1))

        self.pool = nn.MaxPool2d(
            self.pool_size,
            stride=self.pool_size,
            padding=0)

        self.conv1 = nn.Conv2d(
            self.input_channels,
            self.c_out_1,
            self.kernel_size,
            padding=1)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(
            self.conv1.out_channels,
            self.c_out_2,
            self.kernel_size,
            padding=1)

        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(
            self.conv2.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(
            self.conv3.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = nn.Conv2d(
            self.conv4.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)

        self.conv6 = nn.Conv2d(
            self.conv5.out_channels,
            self.c_out_3,
            self.kernel_size,
            padding=1)

        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

        if self.fc_out_h:
            self.fc_out = nn.Linear(self.conv6.out_channels * self.output_height * self.output_width, self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.output_height * self.output_width)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_first(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.conv6(x)))

        x = x.view(-1, self.conv6.out_channels * self.output_height * self.output_width)

        if self.fc_out_h:
            x = self.fc_out(x)

        return x
