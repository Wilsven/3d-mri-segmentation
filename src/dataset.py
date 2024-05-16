import copy
import glob
import json
import os
import tarfile
from abc import ABCMeta

import nibabel as nib
import numpy as np
import torch
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset, random_split

from config import (DATASET_PATH, TASK_ID, TEST_BATCH_SIZE, TRAIN_BATCH_SIZE,
                    TRAIN_VAL_TEST_SPLIT, VAL_BATCH_SIZE)


def ExtractTar(directory: str, dest: str = "./data") -> None:
    try:
        extracted = [
            path
            for path in glob.glob(os.path.join(dest, "*"))
            if not path.endswith(".tar")
        ]
        if extracted:
            print("File is already extracted")
            return

        print("Extracting tar file...")
        tarfile.open(directory).extractall(dest)
    except:
        raise "File extraction failed!"

    print("Extraction completed")
    return


# The dict representing segmentation tasks along with their IDs
task_names = {
    "01": "BrainTumour",
    "02": "Heart",
    "03": "Liver",
    "04": "Hippocampus",
    "05": "Prostate",
    "06": "Lung",
    "07": "Pancreas",
    "08": "HepaticVessel",
    "09": "Spleen",
    "10": "Colon",
}


class MedicalSegmentationDecathlon(Dataset):
    def __init__(
        self,
        task_number: int,
        dir_path: str = "data",
        split_ratios: list[float] = [0.8, 0.1, 0.1],
        transforms: ABCMeta | None = None,
        mode: str | None = None,
    ):
        super().__init__()

        assert task_number in list(
            range(1, 11)
        ), "Task number must be an integer between 1 and 10"

        assert len(split_ratios) == 3, "Split ratios must be of length 3"
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode must be either 'train', 'val' or 'test'"

        # Rectify the task ID representation
        self.task_number = str(task_number)
        if len(self.task_number) == 1:
            self.task_number = "0" + self.task_number

        # Building the file name according to task ID
        self.file_name = f"Task{self.task_number}_{task_names[self.task_number]}"
        # Extracting .tar file
        ExtractTar(os.path.join(dir_path, f"{self.file_name}.tar"))

        # Path to extracted dataset
        self.dir = os.path.join(dir_path, self.file_name)

        # Meta data about the dataset
        self.meta = json.load(open(os.path.join(self.dir, "dataset.json")))
        self.split_ratios = split_ratios
        self.transform = transforms

        # Calculating split number of images
        num_training_imgs = self.meta["numTraining"]
        train_val_test = [int(r * num_training_imgs) for r in split_ratios]
        # If sum of the splits do not match the total number of images, we add the extra to the train split
        if sum(train_val_test) != num_training_imgs:
            extra = num_training_imgs - sum(train_val_test)
            train_val_test[0] += extra

        self.mode = mode

        # Spliting dataset
        samples = self.meta["training"]
        shuffle(samples)
        self.train = samples[: train_val_test[0]]
        self.val = samples[train_val_test[0] : train_val_test[0] + train_val_test[1]]
        self.test = samples[train_val_test[1] : train_val_test[1] + train_val_test[2]]

    def set_mode(self, mode: str):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode must be either 'train', 'val' or 'test'"
        self.mode = mode

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "val":
            return len(self.val)
        elif self.mode == "test":
            return len(self.test)

    def __getitem__(self, idx: int | torch.Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Obtaining image name by given index and the mode using meta data
        if self.mode == "train":
            filename = self.train[idx]["image"].split("/")[-1]
        elif self.mode == "val":
            filename = self.val[idx]["image"].split("/")[-1]
        elif self.mode == "test":
            filename = self.test[idx]["image"].split("/")[-1]

        img_path = os.path.join(self.dir, "imagesTr", filename)
        label_path = os.path.join(self.dir, "labelsTr", filename)

        img_object = nib.load(img_path)
        label_object = nib.load(label_path)

        img_array = img_object.get_fdata()
        label_array = label_object.get_fdata()

        # Converting to channel-first numpy array
        img_array = np.moveaxis(img_array, -1, 0)

        proccessed_out = {
            "filename": filename,
            "image": img_array,
            "label": label_array,
        }

        if self.transform:
            if self.mode == "train":
                proccessed_out = self.transform[0](proccessed_out)
            elif self.mode == "val":
                proccessed_out = self.transform[1](proccessed_out)
            elif self.mode == "test":
                proccessed_out = self.transform[2](proccessed_out)

        # The output numpy array is in channel-first format
        return proccessed_out


def get_train_val_test_data_loaders(
    train_transforms: ABCMeta, val_transforms: ABCMeta, test_transforms: ABCMeta
) -> tuple[DataLoader]:
    dataset = MedicalSegmentationDecathlon(
        task_number=TASK_ID,
        dir_path=DATASET_PATH,
        split_ratios=TRAIN_VAL_TEST_SPLIT,
        transforms=[train_transforms, val_transforms, test_transforms],
    )

    # Spliting dataset and building their respective DataLoaders
    train_set, val_set, test_set = (
        copy.deepcopy(dataset),
        copy.deepcopy(dataset),
        copy.deepcopy(dataset),
    )

    train_set.set_mode("train")
    val_set.set_mode("val")
    test_set.set_mode("test")

    train_dataloader = DataLoader(
        dataset=train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False
    )
    val_dataloader = DataLoader(
        dataset=val_set, batch_size=VAL_BATCH_SIZE, shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_set, batch_size=TEST_BATCH_SIZE, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader
