import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from conf import PATH_ASD
from conf import PATH_TC


def get_data(path: str, label: bool):
    """"
    Load a set of .nii file from a folder
    """
    entries = os.listdir(path)
    datas = []
    labels = []
    for entry in entries:
        if not entry.endswith(".nii"):
            continue
        nib_image = nib.load(path + '\\' + entry).get_fdata()
        datas.append(nib_image)
        labels.append(label)
    train_x = np.array(datas)
    train_y = np.array(labels)
    return train_x, train_y


def load_data():
    x1, y1 = get_data(PATH_ASD, True)
    x2, y2 = get_data(PATH_TC, False)
    train_x = np.concatenate((x1, x2))
    train_y = np.concatenate((y1, y2))
    train_x = np.expand_dims(train_x, axis=1)
    return train_x, train_y
