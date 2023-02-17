# importing libraries
# for creating validation set
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# PyTorch's libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from conf import PATH_TOTAL, ANNOTATION_FILE

from mri_datautils import NiiDataset


class TriConvNet(nn.Module):
    def __init__(self):
        self.net_scale_factor = 1

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
        self.fc1 = nn.Linear(7, self.scale(256))
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
        print(x.shape)
        print("got here!")
        return x

    def scale(self, n1):
        return int(n1 / self.net_scale_factor)


if __name__ == '__main__':
    training_data = NiiDataset(ANNOTATION_FILE, PATH_TOTAL)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4)

    train_features, train_labels = next(iter(train_dataloader))

    net = TriConvNet()
    net = net.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')