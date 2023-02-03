# importing the libraries
import numpy as np
import os

# for creating validation set
from sklearn.model_selection import train_test_split

# PyTorch's libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

from mriutils import load_data


class TriConvNet(nn.Module):
    def __init__(self):
        super(TriConvNet, self).__init__()
        self.conv_layer1 = self._conv_layer_set(1, 32)  # ???
        self.conv_layer2 = self._conv_layer_set(32, 64)  # ???
        ###

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)



    def _conv_layer_set(self, c_in, c_out):
        conv_layer = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



if __name__ == '__main__':
    train_x, train_y = load_data()
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    train_x = torch.from_numpy(train_x)
    val_x = torch.from_numpy(val_x)

    test = TriConvNet()
    test.forward(train_x)
