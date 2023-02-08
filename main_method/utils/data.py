import cv2
import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from torchvision import transforms as TR

import albumentations as A
from albumentations.pytorch import ToTensorV2

import random


def get_img_mask_list(path, csv_type):

    df = pd.read_csv(os.path.join(path, "csv", f"{csv_type}.csv"))
    img_mask_list = df['image_id'].tolist()
    
    img_mask_list = sorted(img_mask_list, key=lambda x: float(x))

    return (img_mask_list, img_mask_list)


def split_data_into_train_val(path_to_train_source, split_ratio):
    
    path_img = os.path.join(os.path.normpath(path_to_train_source), "images")
    path_mask = os.path.join(os.path.normpath(path_to_train_source), "masks")
    img_list = os.listdir(path_img)
    mask_list = os.listdir(path_mask)

    # img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
    img_list = [os.path.splitext(os.path.basename(filename))[0] for filename in img_list if ".png" in filename or ".jpg" in filename]
    # mask_list = [filename for filename in mask_list if ".png" in filename or ".jpg" in filename]
    mask_list = [os.path.splitext(os.path.basename(filename))[0] for filename in mask_list if ".png" in filename or ".jpg" in filename]

    img_list = sorted(img_list, key=lambda x: float(x))
    mask_list = sorted(mask_list, key=lambda x: float(x))  

    img_mask_zipped = list(zip(img_list, mask_list))
    
    random.shuffle(img_mask_zipped)

    img_list_shuffled, mask_list_shuffled = zip(*img_mask_zipped)

    split_index = int(len(img_list_shuffled) * split_ratio)

    image_list_train = img_list_shuffled[:split_index]
    image_list_val = img_list_shuffled[split_index:]
    mask_list_train = mask_list_shuffled[:split_index]
    mask_list_val = mask_list_shuffled[split_index:]

    image_list_train = sorted(image_list_train, key=lambda x: float(x))
    image_list_val = sorted(image_list_val, key=lambda x: float(x))
    mask_list_train = sorted(mask_list_train, key=lambda x: float(x))
    mask_list_val = sorted(mask_list_val, key=lambda x: float(x))

    assert len(image_list_train)  == len(mask_list_train), "different len of images and masks %s - %s" % (len(image_list_train), len(mask_list_train))
    assert len(image_list_val)  == len(mask_list_val), "different len of images and masks %s - %s" % (len(image_list_train), len(mask_list_train))
    
    for i in range(len(image_list_train)):
        assert os.path.splitext(image_list_train[i])[0] == os.path.splitext(mask_list_train[i])[0], '%s and %s are not matching' % (image_list_train[i], mask_list_train[i])

    for i in range(len(image_list_val)):
        assert os.path.splitext(image_list_val[i])[0] == os.path.splitext(mask_list_val[i])[0], '%s and %s are not matching' % (image_list_val[i], mask_list_val[i])

    return image_list_train, mask_list_train, image_list_val, mask_list_val



class DataModuleSegmentation(pl.LightningDataModule):
    def __init__(self, path_to_train_source=None, path_to_train_target=None, use_pseudo_labels=False, path_to_test=None, path_to_predict=None, use_cycle_gan_source=False, load_data_for_tsne=False, coarse_segmentation=None, domain_adaptation=False, pseudo_labels=False, batch_size=16, load_size: int = 256, num_workers=2):
        """
        Args:
            path_to_train_source (string): Path to the folder containing the folder to training images and corresponding training masks - source
            path_to_train_target (string): Path to the folder containing the folder to training images and corresponding training masks - target
            path_to_test (string): Path to the folder containing the folder to test images and corresponding test masks
            coarse_segmentation (int): If int is given, the dataloader of for path_to_train_source (needs to be extended to target, too)
                returns image, coarse_segementation_gt (size: (int,int)), normal_segmentation_gt
            domain_adaptation (bool): 
                True: Ff the DataModuleSegmentation is used as input for a domain_adaptation network (need of source AND target train data), 
                False: Does only need one training dataset (set only path_to_train_source)
            batch_size (int): Batch size used for train, val, test, metrics calculation
            load_size (int): Size to which the images are rescaled - will be squared image
        """
        super().__init__()

        self.domain_adaptation = domain_adaptation
        self.coarse_segmentation = coarse_segmentation

        self.path_to_train_source = path_to_train_source
        self.path_to_train_target = path_to_train_target

        self.path_to_test = path_to_test
        self.path_to_predict = path_to_predict

        self.load_size = load_size
        self.batch_size = batch_size

        self.use_pseudo_labels = use_pseudo_labels

        self.use_cycle_gan_source = use_cycle_gan_source

        self.load_data_for_tsne = load_data_for_tsne

        self.num_workers = num_workers
        if self.path_to_predict is not None:
            self.image_list_predict, self.mask_list_predict = get_img_mask_list(self.path_to_predict, "test")


    def setup(self, stage=None):

        # path_to_predict must only be not None if you want to use a model + DataLoader for inference
        if self.path_to_predict is not None:

            self.predict_data = CustomDataset(image_list=self.image_list_predict, mask_list=self.mask_list_predict, path=self.path_to_predict,
                                                    transfo_for_train=False, load_size=self.load_size, coarse_segmentation=self.coarse_segmentation)

        ## Get image lists for train, val, test for each dataset according to data/csv
        ### Dataset names: CVC-EndoScene, ETIS-LaribPolypDB, KVASIR_SEG
        image_list_train_source, mask_list_train_source = get_img_mask_list(self.path_to_train_source, "train")
        image_list_val_source, mask_list_val_source = get_img_mask_list(self.path_to_train_source, "valid")
        image_list_test_target, mask_list_test_target = get_img_mask_list(self.path_to_test, "test")
        if self.domain_adaptation:
            image_list_train_target, mask_list_train_target = get_img_mask_list(self.path_to_train_target, "valid")

        # Need of a source and target dataset when using a domain_adaptation model
        if self.domain_adaptation:
            # setup source data
            self.train_data_source = CustomDataset(image_list=image_list_train_source, mask_list=mask_list_train_source, path=self.path_to_train_source,
                                                    transfo_for_train=True, load_size=self.load_size, coarse_segmentation=self.coarse_segmentation)
            self.val_data_source = CustomDataset(image_list=image_list_val_source, mask_list=mask_list_val_source, path=self.path_to_train_source, 
                                                transfo_for_train=False, load_size=self.load_size, coarse_segmentation=self.coarse_segmentation)

            # setup target data
            self.train_data_target = CustomDataset(image_list=image_list_train_target, mask_list=mask_list_train_target, path=self.path_to_train_target, 
                                                   use_pseudo_labels=self.use_pseudo_labels, transfo_for_train=False, load_size=self.load_size)
        
            if self.use_cycle_gan_source:
                self.train_data_source_cycle_gan = CustomDataset(image_list=image_list_train_source, mask_list=mask_list_train_source, path=self.path_to_train_source,
                                                                use_cycle_gan_source=True,
                                                                use_pseudo_labels=self.use_pseudo_labels, transfo_for_train=True, load_size=self.load_size)
                self.val_data_source_cycle_gan = CustomDataset(image_list=image_list_val_source, mask_list=mask_list_val_source, path=self.path_to_train_source,
                                                                use_cycle_gan_source=True,
                                                                transfo_for_train=False, load_size=self.load_size, coarse_segmentation=self.coarse_segmentation)

        # Need of only one dataset when NOT using domain_adaptation model (e.g. U-Net) - it is called train_data_source
        else:
            self.train_data_source = CustomDataset(image_list=image_list_train_source, mask_list=mask_list_train_source, path=self.path_to_train_source,
                                                    transfo_for_train=True, load_size=self.load_size, coarse_segmentation=self.coarse_segmentation)
            self.val_data_source = CustomDataset(image_list=image_list_val_source, mask_list=mask_list_val_source, path=self.path_to_train_source, 
                                                transfo_for_train=False, load_size=self.load_size, coarse_segmentation=self.coarse_segmentation) 

        if self.path_to_test is not None:
            self.test_data = CustomDataset(image_list=image_list_test_target, mask_list= mask_list_test_target, path=self.path_to_test, transfo_for_train=False, load_size=self.load_size)

    def train_dataloader(self):

        if self.domain_adaptation:
            loader_source = data.DataLoader(self.train_data_source, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True) # drop_last=True is needed to calculate the FFT for FDA properly
            loader_target = data.DataLoader(self.train_data_target, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True) # drop_last=True is needed to calculate the FFT for FDA properly
            
            if self.use_cycle_gan_source:
                loader_source_cycle_gan = data.DataLoader(self.train_data_source_cycle_gan, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
                loaders = CombinedLoader({"loader_source": loader_source, "loader_target": loader_target, "batch_cycle_gan_source": loader_source_cycle_gan}, mode="max_size_cycle")
            else:
                loaders = CombinedLoader({"loader_source": loader_source, "loader_target": loader_target}, mode="max_size_cycle")

            return loaders
        
        else:
            return data.DataLoader(self.train_data_source, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):

        loader_source = data.DataLoader(self.val_data_source, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

        if self.path_to_test is not None:
            
            loader_target_test = data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

            if self.domain_adaptation:
                
                loader_target_val = data.DataLoader(self.train_data_target, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

                if self.use_cycle_gan_source:
                    loader_source_cycle_gan = data.DataLoader(self.train_data_source_cycle_gan, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
                    loaders = CombinedLoader({"loader_val_source": loader_source, "loader_target_val": loader_target_val, "loader_target_test": loader_target_test, "batch_cycle_gan_source_val": loader_source_cycle_gan}, mode="max_size_cycle")

                else:
                    loaders = CombinedLoader({"loader_val_source": loader_source, "loader_target_val": loader_target_val, "loader_target_test": loader_target_test}, mode="max_size_cycle")

            else:
                loaders = CombinedLoader({"loader_val_source": loader_source, "loader_target_test": loader_target_test}, mode="max_size_cycle")
            
            return loaders

        else:
            return loader_source

    def test_dataloader(self):

        loader_target_test = data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

        return loader_target_test

    def predict_dataloader(self):

        if self.load_data_for_tsne:
            loader_source = data.DataLoader(self.val_data_source, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
            loader_target_val = data.DataLoader(self.train_data_target, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
            loader_predict_data = data.DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            if self.use_cycle_gan_source:
                    loader_source_cycle_gan = data.DataLoader(self.train_data_source_cycle_gan, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)
                    loaders = CombinedLoader({"loader_val_source": loader_source, "loader_target_val": loader_target_val, "batch_cycle_gan_source_val": loader_source_cycle_gan}, mode="max_size_cycle")
            else:
                loaders = CombinedLoader({"loader_val_source": loader_source, "loader_target_val": loader_target_val, "loader_predict_data": loader_predict_data }, mode="max_size_cycle")
            
            return loaders

        else:
            loader_predict = data.DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            return loader_predict


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, mask_list, path, transfo_for_train, use_cycle_gan_source=False, use_pseudo_labels=False, load_size = 256, coarse_segmentation=None):
        """
        Args:
            path (string): Path to the folder containing the folder to training images and corresponding training masks
                the folder need to have the unique names "images" and "masks"
            coarse_segmentation (int): Height and Width of the downsampled segmentation mask
        """

        # Tryout
        self.images = image_list
        self.masks = mask_list
        if use_pseudo_labels:
            print("Using pseudo labels")
            self.m_t = mask_list
            self.paths = (os.path.join(os.path.normpath(path), "images"), os.path.join(os.path.normpath(path), "pseudo_labels"), os.path.join(os.path.normpath(path), "m_t"))
            self.image_postfix, self.mask_postfix, self.m_t_postfix  = self.get_image_and_mask_postfix(self.paths, use_pseudo_labels=True)
        else:
            print("Using NO pseudo labels")
            if use_cycle_gan_source:
                print("Using cycle_gan_imgs")
                self.paths = (os.path.join(os.path.normpath(path), "cycle_gan"), os.path.join(os.path.normpath(path), "masks"))
                self.image_postfix, self.mask_postfix = self.get_image_and_mask_postfix(self.paths)
            else:
                self.paths = (os.path.join(os.path.normpath(path), "images"), os.path.join(os.path.normpath(path), "masks"))
                self.image_postfix, self.mask_postfix = self.get_image_and_mask_postfix(self.paths)

        self.use_pseudo_labels = use_pseudo_labels
        
        self.load_size = load_size

        self.transfo_for_train = transfo_for_train

        self.coarse_segmentation = coarse_segmentation 

        self.resize_transform = A.Resize(self.load_size, self.load_size)
        self.create_coarse_seg_mask = A.Resize(coarse_segmentation, coarse_segmentation)

        self.train_transforms = A.Compose(
            [
                A.ToFloat(),
                # A.MedianBlur(p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.3),
                # A.HueSaturationValue(p=0.4),
                A.Flip(p=0.5),
                # A.Rotate(p=0.3),
                # A.ElasticTransform(p=0.4, alpha_affine=15, interpolation=1, border_mode=0, value=0, mask_value=0),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        print("Train augmentations: ", self.train_transforms)

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

        image = Image.open(os.path.join(os.path.normpath(self.paths[0]), f'{self.images[idx]}{self.image_postfix}')).convert('RGB') # if you get an error in this line, you might have .jpg images
        image = numpy.array(image)
        mask = Image.open(os.path.join(os.path.normpath(self.paths[1]), f'{self.masks[idx]}{self.mask_postfix}')).convert("L") # if you get an error in this line, you might have .jpg images
        mask = numpy.array(mask)

        if self.use_pseudo_labels:
            m_t = Image.open(os.path.join(os.path.normpath(self.paths[2]), f'{self.m_t[idx]}{self.m_t_postfix}')).convert("L") # if you get an error in this line, you might have .jpg images
            m_t = numpy.array(m_t)

            image, pseudo_mask, m_t =  self.apply_transforms(image, mask, m_t=m_t)

            pseudo_mask = pseudo_mask/255

            pseudo_mask = torch.round(pseudo_mask)
            pseudo_mask = pseudo_mask.view([pseudo_mask.size()[0], pseudo_mask.size()[1]]).long()

            m_t = m_t/255
            m_t = torch.round(m_t)
            m_t = m_t.view([m_t.size()[0], m_t.size()[1]]).long()

            return (image, pseudo_mask, m_t)

        else:
            image, mask = self.apply_transforms(image, mask)
            mask = mask/255
            mask = torch.round(mask)
            mask = mask.view([mask.size()[0], mask.size()[1]]).long()

            return (image, mask)

    def get_image_and_mask_postfix(self, paths, use_pseudo_labels=False):
        

        if use_pseudo_labels:
            path_img, path_mask, path_m_t = paths
            m_t_list = os.listdir(path_m_t)
            m_t_postfix = os.path.splitext(os.path.basename(m_t_list[0]))[1]
        else:
            path_img, path_mask = paths
        
        img_list = os.listdir(path_img)
        mask_list = os.listdir(path_mask)

        image_postfix = os.path.splitext(os.path.basename(img_list[0]))[1]
        mask_postfix = os.path.splitext(os.path.basename(mask_list[0]))[1]

        if use_pseudo_labels:
            return image_postfix, mask_postfix, m_t_postfix
        else:
            return image_postfix, mask_postfix



    def apply_transforms(self, image, mask, m_t=None):
        
        if self.load_size is not None:
            if m_t is not None:
                resized_transformed = self.resize_transform(image=image, masks=[mask, m_t])
                image = resized_transformed["image"]
                mask, m_t = resized_transformed["masks"]
            else:
                resized_transformed = self.resize_transform(image=image, mask=mask)
                image = resized_transformed["image"]
                mask = resized_transformed["mask"]

        if self.transfo_for_train:
            if m_t is not None:
                transformed = self.train_transforms(image=image, masks=[mask, m_t])
                image = transformed["image"]
                mask, m_t = transformed["masks"]
            else:
                transformed = self.train_transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

        else:
            if m_t is not None:
                transformed = self.test_transforms(image=image, masks=[mask, m_t])
                image = transformed["image"]
                mask, m_t = transformed["masks"]
            else:
                transformed = self.test_transforms(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

        if m_t is not None:
            return image, mask, m_t

        else:
            return image, mask