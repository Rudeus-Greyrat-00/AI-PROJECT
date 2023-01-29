# importing the libraries
import pandas as pd
import numpy as np
import os

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score

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
        self.conv1 = nn.Conv3d(in_channels=61, out_channels=61, kernel_size=3, stride=1, padding=1)  #



if __name__ == '__main__':
    train_x, train_y = load_data()
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    



