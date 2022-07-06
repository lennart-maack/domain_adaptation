import torch
import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms as TR

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

class DataModuleSegmentation(pl.LightningDataModule):
    def __init__(self, path_to_train_source, path_to_train_target, batch_size, path_to_test = None, load_size = None):
        """
        Args:
            path_to_train (string): Path to the folder containing the folder to training images and corresponding training masks
            path_to_test (string): Path to the folder containing the folder to test images and corresponding test masks
            batch_size (int): Batch size used for train, val, test, metrics calculation
        """
        super().__init__()
        self.path_to_train_source = path_to_train_source
        self.path_to_train_target = path_to_train_target

        self.path_to_test = path_to_test
        self.load_size = load_size
        self.batch_size = batch_size

    def setup(self, stage=None):

        # setup source data
        train_data_source = CustomDataset(self.path_to_train_source, load_size=self.load_size)
        train_set_size = int(len(train_data_source) * 0.8)
        val_set_size = len(train_data_source) - train_set_size
        self.train_data_source, self.val_data_source = data.random_split(train_data_source, [train_set_size, val_set_size])

        # setup target data
        self.train_data_target = CustomDataset(self.path_to_train_target, load_size=self.load_size)
        # train_set_size = int(len(train_data_target) * 0.8)
        # val_set_size = len(train_data_target) - train_set_size
        # self.train_data_target, self.val_data_target = data.random_split(train_data_target, [train_set_size, val_set_size])
        
        if self.path_to_test is not None:
            self.test_data = CustomDataset(self.path_to_test, load_size=self.load_size)

    def train_dataloader(self):

        loader_source = data.DataLoader(self.train_data_source, batch_size=self.batch_size, shuffle=True)
        loader_target = data.DataLoader(self.train_data_target, batch_size=self.batch_size, shuffle=True)

        loaders = CombinedLoader({"loader_source": loader_source, "loader_target": loader_target}, mode="min_size")
        return loaders

    def val_dataloader(self):

        loader_source = data.DataLoader(self.val_data_source, batch_size=self.batch_size)
        # loader_target = data.DataLoader(self.val_data_target, batch_size=self.batch_size)

        # loaders = CombinedLoader({"loader_source": loader_source, "loader_target": loader_target}, mode="min_size")
        return loader_source

    def test_dataloader(self):
        return data.DataLoader(self.val_data, batch_size=self.batch_size)



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, load_size = None):
        """
        Args:
            path (string): Path to the folder containing the folder to training images and corresponding training masks
                the folder need to have the unique names "images" and "masks"
        """
        self.path = path
        self.load_size = load_size
        self.images, self.masks, self.paths = self.list_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.paths[1], self.masks[idx]))
        image, mask = self.transforms(image, mask)
        mask = torch.round(mask)[0]
        mask = mask.view([1, mask.size()[0], mask.size()[1]])

        return (image, mask)

    def list_images(self):
        #mode = "validation" if self.opt.phase == "test" or self.for_metrics else "training"
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

    def transforms(self, image, mask):
        assert image.size == mask.size
        # resize
        if self.load_size is not None:
            new_width, new_height = (self.load_size, self.load_size)
            image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
            mask = TR.functional.resize(mask, (new_width, new_height), Image.NEAREST)
        # to tensor
        image = TR.functional.to_tensor(image)
        mask = TR.functional.to_tensor(mask)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, mask