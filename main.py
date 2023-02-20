""""
This file is the main.py
"""

# importing libraries
# for creating validation set
import os.path

from torch.utils.data import DataLoader

# PyTorch's libraries and modules
import torch
import torch.nn as nn
import torch.optim as optim

from conf import PATH_TOTAL, ANNOTATION_FILE, NET_WEIGHT_PATH

from dataset_utils import make_train_val_datasets, make_cross_validation_datasets

from training import train_and_save, check_accuracy

from models import TriConvNet


def train_test(split_factor: float, net, criterion, optimizer, epochs):
    datasets = make_train_val_datasets(split_factor)
    train_dataloader = DataLoader(datasets[0]["training"], batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(datasets[0]["validation"], batch_size=64, shuffle=True, num_workers=4)

    print("[TRAINING PART]:")

    train_and_save(net=net, criterion=criterion, optimizer=optimizer, epochs=epochs, train_loader=train_dataloader)

    print("[TESTING PART]:")

    check_accuracy(test_dataloader=test_dataloader, net=net)


if __name__ == '__main__':
    net = TriConvNet(inv_scale=2)
    net = net.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_test(split_factor=0.2, net=net, criterion=criterion, optimizer=optimizer, epochs=15)


