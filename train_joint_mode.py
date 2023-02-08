import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import json
import os 

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from main_method.models.model_joint import MainNetwork

from main_method.utils.data import DataModuleSegmentation

import wandb

def main():

    dm = DataModuleSegmentation(path_to_train_source=wandb.config.path_to_train, path_to_train_target=wandb.config.path_to_train_target, domain_adaptation=wandb.config.domain_adaptation,
                                path_to_test=wandb.config.test_data_path, load_size=256,
                                batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers)

    checkpoint_dir = f"{wandb.config.checkpoint_dir}_{wandb.config.run_name}_model_type{wandb.config.model_type}_lr{wandb.config.lr}_contr_lambda_{wandb.config.contrastive_lambda}_contr_head_type_{wandb.config.contr_head_type}_temperature_{wandb.config.temperature}"

    checkpoint_callback = ModelCheckpoint(save_top_k=1, dirpath=checkpoint_dir, monitor="Overall Loss Training", mode="min")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Needed for hyperparameter tuning/sweeps
    print(wandb.config)

    model = MainNetwork(wandb.config)

    trainer = pl.Trainer(callbacks=[checkpoint_callback, lr_monitor], accelerator=wandb.config.device, devices=1, logger=wandb_logger, max_epochs=wandb.config.max_epochs,
                        fast_dev_run=wandb.config.debug)

    wandb_logger.watch(model)

    trainer.fit(model, dm)

    if wandb.config.test_data_path is not None:
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
    parser.add_argument("--visualize_tsne", type=bool, default=True, help="If set to true, tsne is calculated during validation - makes it fast and better for hyperpara opti")
    parser.add_argument("--index_range", type=list, default=[0, 2], help="defines the indices (range from first idx to last idx) that are used for logging/visualizing seg_masks/predictions/feature_maps in a mini batch during validation")

    
    # Arguments for data
    parser.add_argument("--path_to_train", type=str, help="path where the training data is stored")
    parser.add_argument("--path_to_train_target", type=str, help="path where the target training data is stored")
    parser.add_argument("--domain_adaptation", action='store_true', help="If set, domain adaptation is used (a target dataset for training is loaded in data.py)")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--load_size", type=int, default=256, help="size to which images will get resized")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("--num_workers", default=2, type=int, help="num worker for dataloader")


    parser.add_argument("--pretrained_ImageNet", type=bool, default=False, help="If a pretrained ResNet on IMAGENET is used")
    parser.add_argument("--use_self_learning", type=bool, default=False, help="Set to True explicitly if you want to use self-learning with pseudo target labels")
    parser.add_argument("--start_epoch_for_self_learning", default=0, type=int, help="If use_self_learning is true, this is set ")
    parser.add_argument("--ssl_threshold", type=float, default=0.9, help="Threshold confidence value to create pseudo labels")
    parser.add_argument("--use_target_for_contr", type=bool, default=False, help="Set to True explicitly if you want to use target features for contr. loss")
    parser.add_argument("--contrastive_lambda", type=float, default=1.0, help="Coefficient used for the contrastive loss in the overall loss")
    parser.add_argument("--use_confidence_threshold", type=bool, default=False, help="Set to True if you want to use m_t mask to calculate the ce loss for the target data using pseudo masks with confidence")
    

    # Arguments for model and training settings
    parser.add_argument("--model_type", type=str, choices=["normal", "dilated"] , help="What type of backbone is used, either normal or dilated ResNet18")
    parser.add_argument("--apply_FDA", type=bool, default=False, help="Set to True explicitly if you want to use FDA as style transfer step")
    parser.add_argument("--contr_head_type", type=str, default="no_contr_head", choices=["no_contr_head", "contr_head_1"], help="which type of contr head is used to create the feature vector fead into contrastive loss")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--test_after_train", type=bool, help="if set to true and test data provided by test_data_path, the best model will be applied on test data")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
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