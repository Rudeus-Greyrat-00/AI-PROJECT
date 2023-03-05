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

from models import TriConvNet, TriConvNet2


def train_test_split(split_factor: float, net, criterion, optimizer, epochs):
    datasets = make_train_val_datasets(split_factor)  # create the datasets

    # create the dataloaders from the datasets
    train_dataloader = DataLoader(datasets[0]["training"], batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(datasets[0]["validation"], batch_size=64, shuffle=True, num_workers=4)

    print("[TRAINING PART]:")

    #  train the net
    train_and_save(net=net, criterion=criterion, optimizer=optimizer, epochs=epochs, train_loader=train_dataloader)

    print("[TESTING PART]:")

    #  measure the accuracy of the nets
    check_accuracy(test_dataloader=test_dataloader, net=net)


def train_test_cross(k_fold: int, net, criterion, optimizer, epochs):
    datasets = make_cross_validation_datasets(k_fold=k_fold)

    n_of_trainings = len(datasets)
    accuracy_values = []

    for test_train_dict in datasets:
        train_dataloader = DataLoader(test_train_dict["training"], batch_size=64, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_train_dict["validation"], batch_size=64, shuffle=True, num_workers=4)

        train_and_save(net=net, criterion=criterion, optimizer=optimizer, epochs=epochs, train_loader=train_dataloader)

        accuracy = check_accuracy(test_dataloader=test_dataloader, net=net)
        accuracy_values.append(accuracy)

    average_accuracy = sum(accuracy_values) / n_of_trainings
    print(f"Accuracy value measured through cross validation: {average_accuracy}")


if __name__ == '__main__':
    net = TriConvNet2(inv_scale=1)
    net = net.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # stochastic gradient descend

    #train_test_cross(k_fold=10, net=net, criterion=criterion, optimizer=optimizer, epochs=3)
    train_test_split(0.1, net=net, criterion=criterion, optimizer=optimizer, epochs=10)





