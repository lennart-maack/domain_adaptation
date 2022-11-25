import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import json
import os 

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import os
import sys
# Enable module loading from parentfolder
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from basic_UNet.models.UNet import SegModel
from main_method.utils.data import DataModuleSegmentation


import wandb

def main():

    dm = DataModuleSegmentation(path_to_train_source=wandb.config.path_to_train, path_to_test=wandb.config.test_data_path, load_size=256, batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers)

    checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=wandb.config.checkpoint_dir, monitor="Validation Loss (BCE)", mode="min")

    # Needed for hyperparameter tuning/sweeps
    print(wandb.config)

    model = SegModel(model_name=wandb.config.model_name, num_classes=wandb.config.num_classes_for_UNet)

    trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator=wandb.config.device, devices=1, logger=wandb_logger, max_epochs=wandb.config.max_epochs,
                        fast_dev_run=wandb.config.debug)

    wandb_logger.watch(model)

    trainer.fit(model, dm)

    if wandb.config.test_data_path is not None and wandb.config.test_after_train:
        trainer.test(ckpt_path="best", datamodule=dm)
        
        
if __name__ == "__main__":


    parser = ArgumentParser(add_help=False)

    # Read arguments from a json file
    parser.add_argument('--load_json', default=None, help='Path to json file to load settings from file in json format. Command line options override values in file.')

    # Arguments for weights and biases and experiment tracking
    parser.add_argument("--wandb_logging", type=bool, help="if set to false, wandb does not log")
    parser.add_argument("--run_name", type=str, help="Name of the run in wandb")
    parser.add_argument("--project_name", type=str, help="Name of the project in wandb")
    parser.add_argument("--checkpoint_dir", type=str, help="path where the checkpoint (model etc.) should be saved")
    
    
    # Arguments for data
    parser.add_argument("--path_to_train", type=str, help="path where the training data is stored")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--load_size", type=int, default=256, help="size to which images will get resized")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("--num_workers", default=2, type=int, help="num worker for dataloader")


    # Arguments for model and training settings
    parser.add_argument("--model_name", type=str, choices=["unet", "unet_resnet_backbone", "deeplabv3"] , help="What type of backbone is used, either normal or dilated ResNet18")
    parser.add_argument("--num_classes_for_UNet", type=int, help="Number of output channels for the segmentation mask predicted by UNet")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--test_after_train", type=bool, help="if set to true and test data provided by test_data_path, the best model will be applied on test data")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Debugging options
    parser.add_argument("--debug", default=False, type=bool, help="If True, model is run in fast_dev_run (debug mode)")
    parser.add_argument("--overfit_batches", default=0.0, help="Sanity check, takes single batch for training and tries to overfit (Other values can be chosen, check lightning docs")
    hparams = parser.parse_args()

    # create a dictionary from the hparams Namespace
    argparse_dict = vars(hparams)


    if hparams.load_json is not None:
        print("use json file for parameters")
        # open the json file with all the arguments needed for training
        with open(hparams.load_json) as json_file:
            json_dict = json.load(json_file)

        # update the arguments with json content
        argparse_dict.update(json_dict)

        # Create a namespace from the argparse_dict
        hparams = Namespace(**argparse_dict)

    if not hparams.wandb_logging:
        print("Disable wand logging")
        os.environ['WANDB_SILENT']="true"

    wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, log_model="True", config=argparse_dict)

    main()