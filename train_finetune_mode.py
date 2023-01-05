import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import json
import os 
import torch

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from main_method.models.models_pretraining import FineTune, PreTrain

from main_method.utils.data import DataModuleSegmentation

import wandb

from datetime import date

def main():

    dm = DataModuleSegmentation(path_to_train_source=wandb.config.path_to_train, path_to_train_target=wandb.config.path_to_train_target, domain_adaptation=wandb.config.domain_adaptation,
                                use_pseudo_labels=wandb.config.use_pseudo_labels,
                                path_to_test=wandb.config.test_data_path, load_size=256,
                                batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers)

    if wandb.config.contr_pretrain is not None:
        contr_pretrain_dir = os.path.basename(os.path.dirname(wandb.config.contr_pretrain))

        checkpoint_dir = os.path.join(wandb.config.checkpoint_dir, str(date.today()), "Finetune", f"model_{contr_pretrain_dir}_model_type{wandb.config.model_type}_lr{wandb.config.lr}_pretrained_ImageNet{wandb.config.pretrained_ImageNet}")

    else:
        checkpoint_dir = os.path.join(wandb.config.checkpoint_dir, str(date.today()), "Finetune", f"model_NotContrastivlyPretrained_model_type{wandb.config.model_type}_lr{wandb.config.lr}_pretrained_ImageNet{wandb.config.pretrained_ImageNet}")


    checkpoint_callback = ModelCheckpoint(save_top_k=1, dirpath=checkpoint_dir, monitor="Overall Loss Training", mode="min")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Needed for hyperparameter tuning/sweeps
    print(wandb.config)

    if wandb.config.contr_pretrain is not None:
        checkpoint = torch.load(wandb.config.contr_pretrain, map_location=torch.device(wandb.config.device))
        pretrain_model = PreTrain(wandb.config)
        pretrain_model.load_state_dict(checkpoint["state_dict"])
        print("Loaded weights from contrastive pretraining for encoder..")
        encoder = pretrain_model.encoder
        model = FineTune(wandb.config, encoder=encoder, contr_head=pretrain_model.contr_head)
    else:
        model = FineTune(wandb.config)


    if wandb.config.auto_lr_find:
        trainer = pl.Trainer(auto_lr_find=wandb.config.auto_lr_find,
                        callbacks=[checkpoint_callback, lr_monitor], accelerator=wandb.config.device, devices=1, logger=wandb_logger, max_epochs=wandb.config.max_epochs,
                        check_val_every_n_epoch=wandb.config.check_val_every_n_epoch,
                        fast_dev_run=wandb.config.debug)
        lr_finder = trainer.tuner.lr_find(model, dm)
        print()
        print("lr_finder.results: ", lr_finder.results)
        print("lr_finder.suggestion(): ", lr_finder.suggestion())
        print()
        fig = lr_finder.plot(suggest=True)
        fig.show()
        return

    trainer = pl.Trainer(callbacks=[checkpoint_callback, lr_monitor], accelerator=wandb.config.device, devices=1, logger=wandb_logger, max_epochs=wandb.config.max_epochs,
                        check_val_every_n_epoch=wandb.config.check_val_every_n_epoch,
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


    # Arguments for model and training settings
    parser.add_argument("--model_type", type=str, choices=["normal", "dilated"] , help="What type of backbone is used, either normal or dilated ResNet18")

    # Special arguments for finetune mode
    parser.add_argument("--contr_pretrain", type=str, help="Path to the 'Pretrain model' that contains the encoder, used to finetune")
    parser.add_argument("--use_pseudo_labels", type=bool, default=False, help="Set to true if you want to use pseudo_labels for train_target(valid) images - a subfolder to the train_target data with name pseudo_labels needs to excist")
    parser.add_argument("--use_contr_head_for_tsne", type=bool, default=False, help="Set to true if the contr. head should be used on feat embeds before fed into tsne")
    parser.add_argument("--use_confidence_threshold_m_t", type=bool, default=False, help="Set to true if you want to use m_t threshold maps for only using high confidence pixel embeddings for train_target(valid) images - a subfolder to the train_target data with name m_t needs to excist")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2, help="After how many epochs do you want to evaluate")
    parser.add_argument("--auto_lr_find", type=bool, default=False, help="Helps to find the best lr for your model")


    parser.add_argument("--pretrained_ImageNet", type=bool, default=True, help="If a pretrained ResNet on IMAGENET is used")
    parser.add_argument("--apply_FDA", type=bool, default=False, help="Set to True explicitly if you want to use FDA as style transfer step")
    # parser.add_argument("--use_self_learning", type=bool, default=False, help="Set to True explicitly if you want to use self-learning with pseudo target labels")
    # parser.add_argument("--ssl_threshold", type=float, default=0.9, help="Threshold confidence value to create pseudo labels")
    # parser.add_argument("--use_target_for_contr", type=bool, default=True, help="Set to True explicitly if you want to use target features for contr. loss")
    parser.add_argument("--contr_head_type", type=str, default="no_contr_head", choices=["no_contr_head", "contr_head_1"], help="which type of contr head is used to create the feature vector fead into contrastive loss")
    # parser.add_argument("--using_full_decoder", action='store_true', help="If true a normal encoder is used (encoding to original seg mask size, if false no normal encoder is used")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--test_after_train", type=bool, help="if set to true and test data provided by test_data_path, the best model will be applied on test data")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # parser.add_argument("--coarse_lambda", type=float, default=1.0, help="Coefficient used for the coarse loss in the overall loss")
    parser.add_argument("--contrastive_lambda", type=float, default=1.0, help="Coefficient used for the contrastive loss in the overall loss")
    parser.add_argument("--device", default='cuda', type=str, help="device to train on")


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