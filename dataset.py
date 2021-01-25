import torch
import glob
import random
import json
import os

from PIL import Image  # type: ignore
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


class ImageDataset(torch.utils.data.Dataset):
    # glob.glob("data/train_val2019/*/*/*.jpg")
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
