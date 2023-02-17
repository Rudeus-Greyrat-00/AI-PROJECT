import os.path

from torch.utils.data import Dataset
import pandas as pd
from mriutils import read_nii_image


class NiiDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform = None, target_transform = None):
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

