""""
This file contains some utility function for working with dataset
Running this file will test the functions.
"""

import numpy as np

from conf import PATH_TOTAL, ANNOTATION_FILE
from mri_datautils import NiiDataset
from torch.utils.data.dataset import Subset


def make_train_val_datasets(split_factor: float, path=PATH_TOTAL):
    """
    Return a couple of dataset in the form of a list of dictionaries
    [{training : training_dataset, validation : validation_dataset}]
    You may wonder, but why returning a dictionary inside a one-sized list? Well, this way it is consistent with the
    return value of the other function (make_cross_validation_datasets), which is great isn't it?
    :param split_factor:
    :param path:
    :return:
    """
    return_list = []

    all_data = NiiDataset(annotation_file=ANNOTATION_FILE, img_dir=PATH_TOTAL)

    select_v = np.ones(len(all_data))
    select_v[0:int(len(select_v) * split_factor)] = 0

    np.random.shuffle(select_v)

    train_indexes = np.where(select_v == 1)[0]
    test_indexes = np.where(select_v == 0)[0]

    train_set = all_data.subset(train_indexes)
    val_set = all_data.subset(test_indexes)

    return_list.append({"training": train_set, "validation": val_set})

    return return_list


def make_cross_validation_datasets(k_fold: int, path=PATH_TOTAL):
    """
    Return a bunch of dataset in the form of a list of dictionaries
    [{training : training_dataset, validation : validation_dataset},
    ... repeated k_fold times ...
    {training : training_dataset, validation : validation_dataset}]
    :param k_fold:
    :param path:
    :return:
    """

    return_list = []

    all_data = NiiDataset(annotation_file=ANNOTATION_FILE, img_dir=PATH_TOTAL)

    total_size = len(all_data)

    fraction = 1 / k_fold
    seg = int(total_size * fraction)  # segments length

    for i in range(k_fold):
        """"
        Example:
        [a b c d e f g h i l m n o p q r s t u v z] → the dataset
        [↑ train portion ↑ val ↑ train portion   ↑] → the indexes used for cross validation (with k = 4 in this case)
                ↑ train left          ↑ train right
        """

        train_left_start = 0
        train_left_end = i * seg
        validation_start = train_left_end
        validation_end = i * seg + seg
        train_right_start = validation_end
        train_right_end = total_size

        train_left_indices = list(range(train_left_start, train_left_end))
        train_right_indices = list(range(train_right_start, train_right_end))

        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(validation_start, validation_end))

        train_set = all_data.subset(train_indices)
        val_set = all_data.subset(val_indices)

        return_list.append({"training": train_set, "validation": val_set})

    return return_list


if __name__ == '__main__':
    print("Testing functions...")
    make_train_val_datasets(split_factor=0.2)
    make_cross_validation_datasets(k_fold=10)

