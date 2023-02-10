# importing libraries
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
        self.net_scale_factor = 2

        super(TriConvNet, self).__init__()
        self.conv_layer1 = self._layer_conv(1, self.scale(32), (2, 2, 2))
        self.conv_layer2 = self._layer_conv_maxpool(self.scale(32), self.scale(64), (1, 1, 1))
        # self.conv_layer3 = self._layer_conv(self.scale(64), self.scale(128), (2, 2, 2))
        # self.conv_layer4 = self._layer_conv(self.scale(128), self.scale(256), (1, 1, 1))
        self.conv_layer5 = self._layer_conv(self.scale(64), self.scale(256), (2, 2, 2))
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer6 = self._layer_conv(self.scale(256), self.scale(128), (1, 1, 1))
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_layer7 = self._layer_conv(self.scale(128), self.scale(64), (2, 2, 2))
        self.conv_layer8 = nn.Conv3d(self.scale(64), self.scale(32), kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.layer9 = nn.ReLU()
        self.fc1 = nn.Linear(7, self.scale(256))  #TODO check input qui
        self.fc2 = nn.Linear(self.scale(256), self.scale(256))
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

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        #x = self.conv_layer3(x)
        #x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.upsample1(x)
        x = self.conv_layer6(x)
        x = self.upsample2(x)
        x = self.conv_layer7(x)
        x = self.conv_layer8(x)
        x = self.layer9(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def scale(self, n1):
        return int(n1 / self.net_scale_factor)


if __name__ == '__main__':
    train_x, train_y = load_data()  # load data, train_x and train_y are numpy array. y -> labels. x -> datas
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.05)  # split data
    train_x = torch.from_numpy(train_x)  # it transforms the inputs from numpy array to pytorch tensors
    val_x = torch.from_numpy(val_x)

    net = TriConvNet()  # create an instance of the net
    net = net.double()

    #  test

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
