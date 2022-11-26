import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import json
import os 

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from main_method.models.main_model import MainNetwork

from main_method.utils.data import DataModuleSegmentation


import wandb

def main():

    dm = DataModuleSegmentation(path_to_train_source=wandb.config.path_to_train, path_to_test=wandb.config.test_data_path, coarse_segmentation=wandb.config.coarse_segmentation_size, load_size=256, batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers)

    if wandb.config.using_full_decoder:
        checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=wandb.config.checkpoint_dir, monitor="Validation Loss (BCE)", mode="min")
    elif wandb.config.coarse_prediction_type != "no_coarse" and wandb.config.use_coarse_outputs_for_contrastive:
        checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=wandb.config.checkpoint_dir, monitor="Validation Loss (BCE)", mode="min")
    else:
        raise NotImplementedError("No correct metric to monitor for checkpoint callback implemented")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Needed for hyperparameter tuning/sweeps
    print(wandb.config)

    model = MainNetwork(wandb.config)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, lr_monitor], accelerator=wandb.config.device, devices=1, logger=wandb_logger, max_epochs=wandb.config.max_epochs,
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
    parser.add_argument("--visualize_tsne", action='store_true', help="If true, no tsne is calculated during validation - makes it fast and better for hyperpara opti")
    parser.add_argument("--index_range", type=list, default=[0, 6], help="defines the indices (range from first idx to last idx) that are used for logging/visualizing seg_masks/predictions/feature_maps in a mini batch during validation")

    
    # Arguments for data
    parser.add_argument("--path_to_train", type=str, help="path where the training data is stored")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--load_size", type=int, default=256, help="size to which images will get resized")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("--num_workers", default=2, type=int, help="num worker for dataloader")


    # Arguments for model and training settings
    parser.add_argument("--model_type", type=str, choices=["normal", "dilated"] , help="What type of backbone is used, either normal or dilated ResNet18")
    parser.add_argument("--coarse_prediction_type", type=str, default="no_coarse", choices=["no_coarse", "linear", "mlp"], help="which type of coarse prediction should be used")
    parser.add_argument("--coarse_segmentation_size", type=int, default=None, help="Size of the coarse segmentation after the dilated backbone - this is only important to load the train, val and test images in the correct size for evaluation")
    parser.add_argument("--contr_head_type", type=str, default="no_contr_head", choices=["no_contr_head", "contr_head_1"], help="which type of contr head is used to create the feature vector fead into contrastive loss")
    parser.add_argument("--using_full_decoder", action='store_true', help="If true a normal encoder is used (encoding to original seg mask size, if false no normal encoder is used")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--test_after_train", type=bool, help="if set to true and test data provided by test_data_path, the best model will be applied on test data")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--coarse_lambda", type=float, default=1.0, help="Coefficient used for the coarse loss in the overall loss")
    parser.add_argument("--contrastive_lambda", type=float, default=1.0, help="Coefficient used for the contrastive loss in the overall loss")
    parser.add_argument("--device", default='gpu', type=str, help="device to train on")


    # Arguments for contrastive learning
    parser.add_argument("--temperature", type=float, default=0.3, help="temperature for contrastive loss")
    parser.add_argument("--base_temperature", type=float, default=0.07, help="temperature for contrastive loss")
    parser.add_argument("--max_samples", type=int, default=2048, help="max_samples for contrastive loss")
    parser.add_argument("--max_views", type=int, default=100, help="max_views for contrastive loss")
    parser.add_argument("--use_coarse_outputs_for_contrastive", action='store_true', help="If used, the coarse output of the \
        segmentation model (prediction and mask) is used as the input to the contrastive loss. If not the full segmentation mask and prediction is downsampled to \
            H* and W* of the feature embedding")
    
    
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