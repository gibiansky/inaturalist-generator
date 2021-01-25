import unittest
import torchvision # type: ignore
import torch

import dataset

class TestShittyDataset(unittest.TestCase):
    def test_sanity(self):
        val_images = dataset.ShittyDataset("val2019.json", base_dir="./data",
            transforms=torchvision.transforms.RandomCrop([10, 10]), seed=1)

        ## do we get all the images and crop them? hopefully
        img = val_images[0]
        self.assertEqual(list(img.shape), [3, 10, 10])
        self.assertEqual(len(val_images), 3030)

        val_images_2 = dataset.ShittyDataset("val2019.json", base_dir="./data",
            transforms=torchvision.transforms.RandomCrop([10, 10]), seed=2)
        img2 = val_images_2[0]

        # are we shuffling?
        self.assertFalse(torch.equal(img, img2))


if __name__ == '__main__':
    unittest.main()
