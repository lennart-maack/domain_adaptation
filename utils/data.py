import cv2
import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from torchvision import transforms as TR

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataModuleSegmentation(pl.LightningDataModule):
    def __init__(self, path_to_train_source=None, path_to_train_target=None, path_to_test=None, domain_adaptation=False, batch_size=16, load_size=None):
        """
        Args:
            path_to_train_source (string): Path to the folder containing the folder to training images and corresponding training masks - source
            path_to_train_target (string): Path to the folder containing the folder to training images and corresponding training masks - target
            path_to_test (string): Path to the folder containing the folder to test images and corresponding test masks
            domain_adaptation (bool): If the DataModuleSegmentation is used as input for a domain_adaptation network (need of source
                AND target train data, whereas domain_adaptation=False does only need one training dataset (either target or train))
            batch_size (int): Batch size used for train, val, test, metrics calculation
            load_size (tuple:(int,int)): Size to which the images are rescaled
        """
        super().__init__()

        self.domain_adaptation = domain_adaptation

        self.path_to_train_source = path_to_train_source
        self.path_to_train_target = path_to_train_target

        self.path_to_test = path_to_test
        self.load_size = load_size
        self.batch_size = batch_size

    def setup(self, stage=None):

        # Need of a source and target dataset when using a domain_adaptation model
        if self.domain_adaptation:
            # setup source data
            train_data_source = CustomDataset(self.path_to_train_source, transfo_for_train=True, load_size=self.load_size)
            train_set_size = int(len(train_data_source) * 0.8)
            val_set_size = len(train_data_source) - train_set_size
            self.train_data_source, self.val_data_source = data.random_split(train_data_source, [train_set_size, val_set_size])

            # setup target data
            self.train_data_target = CustomDataset(self.path_to_train_target, transfo_for_train=True, load_size=self.load_size)
        
        # Need of only one dataset when NOT using domain_adaptation model (e.g. U-Net) - it is called train_data_source
        else:
            train_data_source = CustomDataset(self.path_to_train_source, transfo_for_train=True, load_size=self.load_size)
            train_set_size = int(len(train_data_source) * 0.8)
            val_set_size = len(train_data_source) - train_set_size
            self.train_data_source, self.val_data_source = data.random_split(train_data_source, [train_set_size, val_set_size])   

        if self.path_to_test is not None:
            self.test_data = CustomDataset(self.path_to_test, transfo_for_train=False, load_size=self.load_size)

    def train_dataloader(self):

        if self.domain_adaptation:
            loader_source = data.DataLoader(self.train_data_source, batch_size=self.batch_size, shuffle=True, num_workers=2)
            loader_target = data.DataLoader(self.train_data_target, batch_size=self.batch_size, shuffle=True, num_workers=2)

            loaders = CombinedLoader({"loader_source": loader_source, "loader_target": loader_target}, mode="min_size")
            return loaders
        
        else:
            return data.DataLoader(self.train_data_source, batch_size=self.batch_size, shuffle=True, num_workers=2)


    def val_dataloader(self):

        loader_source = data.DataLoader(self.val_data_source, batch_size=self.batch_size, num_workers=2)

        loader_target_test = data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2)

        loaders = CombinedLoader({"loader_val_source": loader_source, "loader_target_test": loader_target_test}, mode="max_size_cycle")
        
        return loaders

    def test_dataloader(self):

        loader_target_test = data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2)

        return loader_target_test


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, transfo_for_train, load_size = None):
        """
        Args:
            path (string): Path to the folder containing the folder to training images and corresponding training masks
                the folder need to have the unique names "images" and "masks"
        """
        self.path = path
        self.load_size = load_size
        self.images, self.masks, self.paths = self.list_images()
        self.transfo_for_train = transfo_for_train

        self.resize_transform = A.Resize(self.load_size, self.load_size)

        self.train_transforms = A.Compose(
            [
                A.ToFloat(),
                # A.MedianBlur(p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=0.7),
                # A.HueSaturationValue(p=0.5),
                A.Flip(p=0.6),
                # # A.Rotate(p=0.5),
                # A.ElasticTransform(p=0.3),
                # # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.test_transforms = A.Compose(
            [
                A.ToFloat(),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        ) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        image = np.array(image)
        mask = Image.open(os.path.join(self.paths[1], self.masks[idx])).convert("L")
        mask = np.array(mask)

        image, mask = self.apply_transforms(image, mask)
        mask = mask/255
        mask = torch.round(mask)
        mask = mask.view([1, mask.size()[0], mask.size()[1]])

        return (image, mask)

    def list_images(self):

        path_img = os.path.join(self.path, "images") #, mode)
        path_mask = os.path.join(self.path, "masks") #, mode)
        img_list = os.listdir(path_img)
        mask_list = os.listdir(path_mask)
        img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
        mask_list = [filename for filename in mask_list if ".png" in filename or ".jpg" in filename]
        images = sorted(img_list)
        masks = sorted(mask_list)
        assert len(images)  == len(masks), "different len of images and masks %s - %s" % (len(images), len(masks))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(masks[i])[0], '%s and %s are not matching' % (images[i], masks[i])
        return images, masks, (path_img, path_mask)

    def apply_transforms(self, image, mask):
                
        if self.load_size is not None:
            resized_transformed = self.resize_transform(image=image, mask=mask)
            image = resized_transformed["image"]
            mask = resized_transformed["mask"]

        if self.transfo_for_train:
            transformed = self.train_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        else:
            transformed = self.test_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask