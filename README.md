# domain_adaptation

Repository for domain adaptation. 

This repository contains:
1. "main_method" folder with new domain adaptation approach using contrastive learning
2. "other_domain_adapt_methods" folder with SOTA methods for domain adaptation
3. "basic_UNet" folder with baseline segmentation networks, such as UNet, UNet with ResNet18 backbone, DeepLabv3


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

## Train a Network

To train a Network, the train.py file reads the arguments.json file for setting parameters.
The parameters are described in the corresponding train.py (or train_UNet.py in /basic_UNet).

To start training with the main_method, set the arguments.json and execute the following:

```
python "/path/to/train.py" --load_json "/path/to/parameters.json" 

```

### Train a UNet

If you want to train a UNet or UNet with ResNet18 backbone, there are two ways.

1. Use basic_UNet/train_UNet.py and set the corresponding parameters in arguments_UNet.json (model_type, num_classes etc.)
2. You could also se train.py and set the following parameters (model_type: "normal", coarse_prediction_type: "no_coarse", contr_head_type: "no_contr_head", using_full_decoder: true)