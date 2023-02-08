# domain_adaptation

Repository for "Unsupervised Domain Adaptation for Colon Polyp Segmentation using Pixel Level Contrastive Learning." 

This repository contains the code for:
1. Pretraining with supervised pixel contrastive learning (train_pretrain_mode.py)
2. Finetuning after contrastive pretraining (train_joint_mode.py)
3. Joint training using supervised pixel contrastive loss and pixel cross entropy loss (train_joint_mode.py)
4. Other state-of-the-art methods (refer to folder other_SOTAs):
    1. U-Net baseline (basic_UNet)
    2. Domain Adaptation Neural Network (DANN)
    3. Fourier Domain Adaptation method (FDA)
    4. Contrastive Fourier Domain Adaptation (CoFo)
    5. Mutual Prototype Alignment for Domain Adaptation (MPA-DA)


## Installation

Clone this repository
```
git clone https://github.com/lennart-maack/domain_adaptation.git
```

## Requirements
Run the following command to install all needed requirements. At best, create a virtual environment for the project.

```
pip install -r requirements.txt
```

## Dataset

You should structure your dataset (train and test) in the following way:

```
dataset/
    ├── images/
        ├── name_of_image_1.png
        ├── name_of_image_2.png
        ...
    ├── masks/
        ├── name_of_image_1.png
        ├── name_of_image_2.png
        ...
```

## Train the proposed framework

To train a Network, use the train_pretrain_mode.py, train_finetune_mode.py or train_joint_mode.py file with the corresponding arguments. Details about usable arguments can be found in each of the three files.

## Train a state-of-the-art method

Refer to the README of the corresponding SOTA method folder. 