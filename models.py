"""
This file contains the 3D cnn models.
Actually there is only one
Running this file nothing remarkable should happen
"""

import torch.nn as nn


class TriConvNet(nn.Module):
    def __init__(self, inv_scale: int = 1):
        self.net_scale_factor = 8

        super(TriConvNet, self).__init__()
        self.conv_layer1 = self._layer_conv(1, self.scale(32), (2, 2, 2))
        self.conv_layer2 = self._layer_conv_maxpool(self.scale(32), self.scale(64), (1, 1, 1))
        # self.conv_layer3 = self._layer_conv(self.scale(64), self.scale(128), (2, 2, 2))
        # self.conv_layer4 = self._layer_conv(self.scale(128), self.scale(256), (1, 1, 1))
        self.conv_layer5 = self._layer_conv(self.scale(64), self.scale(256), (2, 2, 2))
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer6 = self._layer_conv(self.scale(256), self.scale(128), (1, 1, 1))
        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer7 = self._layer_conv(self.scale(128), self.scale(64), (2, 2, 2))
        self.conv_layer8 = nn.Conv3d(self.scale(64), self.scale(32), kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.layer9 = nn.ReLU()
        self.fc1 = self._layer_fully_connected(self.scale(17248), self.scale(256))
        self.fc2 = self._layer_fully_connected(self.scale(256), self.scale(256))
        self.fc3 = nn.Linear(self.scale(256), 2)

    def _layer_conv_maxpool(self, c_in, c_out, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3), stride=stride),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def _layer_conv(self, c_in, c_out, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3), stride=stride),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU()
        )
        return conv_layer

    def _layer_fully_connected(self, c_in, c_out):
        layer = nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
            nn.Dropout(),
        )
        return layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        # x = self.conv_layer3(x)
        # x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.up_sample_1(x)
        x = self.conv_layer6(x)
        x = self.up_sample_2(x)
        x = self.conv_layer7(x)
        x = self.conv_layer8(x)
        x = self.layer9(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def scale(self, n1):
        return int(n1 / self.net_scale_factor)
