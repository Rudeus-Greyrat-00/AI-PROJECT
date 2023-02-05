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
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 13 * 16 * 13, 12)
        self.fc2 = nn.Linear(12, 2)



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
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



if __name__ == '__main__':

    train_x, train_y = load_data()
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.05)
    train_x = torch.from_numpy(train_x)
    val_x = torch.from_numpy(val_x)

    net = TriConvNet()
    net = net.double()

    val_x = net.forward(val_x)

    """


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(10):
        for i in range(len(val_x)):
            inputs = val_x[i]
            labels = val_y[i]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    """
