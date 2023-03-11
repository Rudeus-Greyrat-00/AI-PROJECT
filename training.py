""""
This file contains both training and accuracy test functions
Running this file does nothing
"""

import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn


def train_and_save(net: nn.Module, criterion, optimizer, epochs: int, train_loader: DataLoader, save_folder: str = None,
                   filename: str = None):
    print("Starting training...")

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"\tRunning epoch {epoch + 1}/{epochs}:")
        for i, data in enumerate(train_loader, 0):
            print(f"\t\tEpoch {epoch + 1}, batch {i + 1}:")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print("\t\tDone!")
        print(f"\tEpoch {epoch + 1} done!")

    if filename is None or save_folder is None:
        print("Not saving since filename of save_folder are None")
        return 0

    torch.save(net.state_dict(), os.path.join(save_folder, filename))
    print(f"Result (weights) saved as {os.path.join(save_folder, filename)}")

    return os.path.join(save_folder, filename)


def check_accuracy(test_dataloader, net, load_state=False, load_path=None):
    if load_state:
        net.load_state_dict(torch.load(load_path))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:  # batches
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total

    print(f'Accuracy of the network on test images: {accuracy} %')

    return accuracy

