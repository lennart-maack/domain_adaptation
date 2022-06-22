import torch
import torch.utils.data as data
from PIL import Image
import os

import pytorch_lightning as pl

class DataModuleSegmentation(pl.LightningDataModule):
    def __init__(self, path_to_train, path_to_test = None, batch_size = 32):
        """
        Args:
            path_to_train (string): Path to the folder containing the folder to training images and corresponding training masks
            path_to_train (string): Path to the folder containing the folder to test images and corresponding test masks
            batch_size: 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.batch_size = batch_size

    def setup(self):

        train_data = CustomDataset(self.path_to_train)
        train_set_size = int(len(train_data) * 0.8)
        val_set_size = len(train_data) - train_set_size
        self.train_data, self.val_data = data.random_split(train_data, [train_set_size, val_set_size])
        
        self.test_data = CustomDataset(self.path_to_test)

    def train_dataloader(self):
        return data.DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return data.DataLoader(self.val_data, batch_size=self.batch_size)





class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        """
        Args:
            path (string): Path to the folder containing the folder to training images and corresponding training masks
                the folder need to have the unique names "images" and "masks"
        """
        self.path_to_images = os.path.join(path, "images")
        self.path_to_masks = os.path.join(path, "masks")

    def __len__(self):
        len_images = len([image for image in os.listdir(self.path_to_images)])
        len_masks = len([image for image in os.listdir(self.path_to_masks)])
        assert len_images == len_masks, "The number of images is not equal to the number of the corresponding masks"
        return len_images

    def __getitem__(self):
        image = Image.open(os.path.join(self.path_to_images)).convert('RGB')
        mask = Image.open(os.path.join(self.paths[1]))

        return (image, mask)