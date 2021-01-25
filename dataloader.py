from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision  # type: ignore

import dataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config: Any, base_dir: str, height: int, width: int) -> None:
        """
        base_dir is probably ./data
        height and width are up to your own artistic inspiration
        """
        super().__init__()
        self.train_images = dataset.ImageDataset(
            "train2019.json",
            base_dir=base_dir,
            transforms=torchvision.transforms.RandomCrop([height, width], None, True),
        )
        self.val_images = dataset.ImageDataset(
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
