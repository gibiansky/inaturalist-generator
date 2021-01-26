from typing import Any
import glob
import json
import os
import random

from PIL import Image  # type: ignore
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import torchvision  # type: ignore

"""
train_val2019.tar.gz - Contains the training and validation images in a
directory structure following 
    {iconic category name}/{category name}/{image id}.jpg 
    
for example
    "train_val2019/Insects/43/243e5d9fb75b6df813c0c1918ea97869.jpg"

What kind of category is 43? Who knows. It's a mysterious category of insects,
maybe only COOL ONES.

train2019.json - Contains the training annotations.
val2019.json - Contains the validation annotations.
"""


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config: Any, base_dir: str, height: int, width: int) -> None:
        """
        base_dir is probably ./data
        height and width are up to your own artistic inspiration
        """
        super().__init__()
        self.train_images = ImageDataset(
            "train2019.json",
            base_dir=base_dir,
            transforms=torchvision.transforms.RandomCrop([height, width], None, True),
        )
        self.val_images = ImageDataset(
            "val2019.json",
            base_dir=base_dir,
            transforms=torchvision.transforms.CenterCrop([height, width]),
        )
        self.config: Any = config

    def train_dataloader(self):
        return DataLoader(
            self.train_images, batch_size=self.config.data.batch_size, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_images, batch_size=self.config.data.batch_size, num_workers=8
        )


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, transforms, base_dir="./data", seed=0):
        """
        annotations is train2019.json or val2019.json
        base_dir is probably ./data
        seed is whatever your heart desires
        """

        super().__init__()

        with open(os.path.join(base_dir, annotations_file)) as f:
            annotations = json.load(f)

        self.images = [
            os.path.join(base_dir, image["file_name"])
            for image in annotations["images"]
        ]
        self.transforms = torchvision.transforms.Compose(
            [
                transforms,
                torchvision.transforms.ToTensor(),
            ]
        )
        random.Random(seed).shuffle(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")

        with torch.no_grad():
            return self.transforms(image)

    def __len__(self):
        return len(self.images)


class QuantizedDataModule(pl.LightningDataModule):
    def __init__(
        self, config: Any, base_dir: str, quantized_dir: str, height: int, width: int
    ) -> None:
        """
        base_dir is probably ./data
        height and width are up to your own artistic inspiration
        """
        super().__init__()
        self.train_images = QuantizedDataset(
            "train2019.json",
            base_dir=base_dir,
            quantized_dir=quantized_dir,
            transforms=torchvision.transforms.RandomCrop(
                [height, width], None, True, padding_mode="reflect"
            ),
        )
        self.val_images = QuantizedDataset(
            "val2019.json",
            base_dir=base_dir,
            quantized_dir=quantized_dir,
            transforms=torchvision.transforms.CenterCrop([height, width]),
        )
        self.config: Any = config

    def train_dataloader(self):
        return DataLoader(
            self.train_images, batch_size=self.config.data.batch_size, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_images, batch_size=self.config.data.batch_size, num_workers=8
        )


class QuantizedDataset(torch.utils.data.Dataset):
    def __init__(
        self, annotations_file, quantized_dir, transforms, base_dir="./data", seed=0
    ):
        """
        annotations is train2019.json or val2019.json
        base_dir is probably ./data
        seed is whatever your heart desires
        """

        super().__init__()

        with open(os.path.join(base_dir, annotations_file)) as f:
            annotations = json.load(f)

        image_id_to_category_id = {
            annotation["image_id"]: annotation["category_id"]
            for annotation in annotations["annotations"]
        }
        self.npy_files = [
            os.path.join(quantized_dir, image["file_name"].replace(".jpg", ".npy"))
            for image in annotations["images"]
        ]
        self.category_ids = [
            image_id_to_category_id[image["id"]]
            for image in annotations["images"]
        ]
        self.num_categories = max(self.category_ids) + 1
        self.transforms = transforms
        random.Random(seed).shuffle(self.npy_files)

    def __getitem__(self, idx):
        array = np.load(self.npy_files[idx])
        category = self.category_ids[idx]
        assert 0 <= category <= self.num_categories

        with torch.no_grad():
            tensor = torch.LongTensor(array)
            return category, self.transforms(tensor)

    def __len__(self):
        return len(self.npy_files)
