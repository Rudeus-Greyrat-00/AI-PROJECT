import os.path
import os

import torch
from torch.utils.data import Dataset
import pandas as pd

import nibabel as nib
import numpy as np

import csv

""""
This file contains utility class definitions to work with pytorch using .nii files
"""


def read_nii_image(path: str):
    """
    Take the path of a nii image, read it and return a pytorch tensor representing the image.
    :param path:
    :return:
    """
    nib_image = nib.load(path).get_fdata()
    np_arr = np.array(nib_image)
    np_arr = np.expand_dims(np_arr, axis=0)  # add one dimension for the channel
    return torch.from_numpy(np_arr)


def generate_label(path_ASD: str, path_TC: str, path_TOTAL: str):
    """"
    Given a folder of .nii image of people affected by autism (path_ASD) AND a folder of .nii image of people NOT
    affected by autism (path_TC) and a destination folder (path_TOTAL) it generate the label file necessary to the
    dataloader and dataset classes of pytorch
    """
    csv_path = path_TOTAL + "\\" + "label.csv"
    with open(csv_path, "w") as file:
        writer = csv.writer(file)
        entries = os.listdir(path_TC)
        for entry in entries:
            if not entry.endswith(".nii"):
                continue
            writer.writerow([entry, 0])
        entries = os.listdir(path_ASD)
        for entry in entries:
            if not entry.endswith(".nii"):
                continue
            writer.writerow([entry, 1])


if __name__ == '__main__':
    from conf import PATH_TOTAL, PATH_TC, PATH_ASD

    generate_label(path_ASD=PATH_ASD, path_TC=PATH_TC, path_TOTAL=PATH_TOTAL)


class NiiDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_label = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_trasnform = target_transform

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_label.iloc[idx, 0])
        image = read_nii_image(img_path)
        label = self.img_label.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_trasnform:
            label = self.target_trasnform(label)
        return image, label
