import os

import nibabel as nib
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset

import csv


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


def load_data(get_from_exsample : bool = False):
    if(get_from_exsample):
        x1, y1 = get_data(".\\example\\ASD", True)
        x2, y2 = get_data(".\\example\\TC", False)
    else:
        from conf import PATH_ASD
        from conf import PATH_TC
        x1, y1 = get_data(PATH_ASD, True)
        x2, y2 = get_data(PATH_TC, False)
    train_x = np.concatenate((x1, x2))
    train_y = np.concatenate((y1, y2))
    train_x = np.expand_dims(train_x, axis=1)
    return train_x, train_y

def read_nii_image(path : str):
    nib_image = nib.load(path).get_fdata()
    nparr = np.array(nib_image)
    nparr = np.expand_dims(nparr, axis=0)  # channel
    return torch.from_numpy(nparr)

def generate_label():
    from conf import PATH_ASD
    from conf import PATH_TC
    from conf import PATH_TOTAL
    csv_path = PATH_TOTAL + "\\" + "label.csv"
    with open(csv_path, "w") as file:
        entries = os.listdir(PATH_TC)
        writer = csv.writer(file)
        entries = os.listdir(PATH_TC)
        for entry in entries:
            if not entry.endswith(".nii"):
                continue
            writer.writerow([entry, 0])
        entries = os.listdir(PATH_ASD)
        for entry in entries:
            if not entry.endswith(".nii"):
                continue
            writer.writerow([entry, 1])

if __name__ == '__main__':
    generate_label()


