import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from conf import PATH_ASD
from conf import PATH_TC

class LabeledData:
    def __init__(self, name: str, data, label: bool):
        self.name = name
        self.data = data
        self.label = label

#hello
def get_data(path: str, label: bool):
    """"
    Load a set of .nii file from a folder, return a list of tridimensional matrixes
    """
    entries = os.listdir(path)
    datas = []
    for entry in entries:
        if not entry.endswith(".nii"):
            continue
        nib_image = nib.load(path + '\\' + entry).get_fdata()
        datas.append(LabeledData(entry, nib_image, label))
        print(f"Image {entry} loaded, dimension = {nib_image.shape}")
    print("LOAD COMPLETED")
    return datas

#fjkhaskfals



ASD_DATA = get_data(PATH_ASD, True)
TC_DATA = get_data(PATH_TC, False)

DATA = ASD_DATA + TC_DATA
