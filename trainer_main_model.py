import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import json

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.main_model import MainNetwork

from utils.data import DataModuleSegmentation

import wandb


def main(hparams: Namespace):

    dm = DataModuleSegmentation(path_to_train_source=hparams.path_to_train, load_size=256, coarse_segmentation=33, num_workers=hparams.num_workers)

    checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=hparams.checkpoint_dir, monitor="Coarse Dice Score (Validation)", mode="max")

    wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, log_model="True")

    model = MainNetwork(index_range=hparams.index_range, model_type=hparams.model_type, coarse_prediction_type=hparams.coarse_prediction_type, 
                        coarse_lambda=hparams.coarse_lambda, contr_head_type=hparams.contr_head_type, using_full_decoder=hparams.using_full_decoder)

    trainer = pl.Trainer(callbacks=checkpoint_callback, accelerator=hparams.device, devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs,
                        fast_dev_run=hparams.debug)

    wandb_logger.watch(model)

    trainer.fit(model, dm)

    if hparams.test_data_path is not None:
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--run_name", type=str, help="Name of the run in wandb")
    parser.add_argument("--project_name", type=str, help="Name of the project in wandb")
    parser.add_argument("--checkpoint_dir", type=str, help="path where the checkpoint (model etc.) should be saved")
    parser.add_argument("--path_to_train", type=str, help="path where the training data is stored")
    parser.add_argument("--model_type", type=str, choices=["normal", "dilated"] , help="What type of model is used, either normal or dilated ResNet18")
    parser.add_argument("--coarse_prediction_type", type=str, choices=["no_coarse", "linear", "mlp"], help="which type of coarse prediction should be used")
    parser.add_argument("--contr_head_type", type=str, default="no_contr_head", choices=["no_contr_head", "contr_head_1", "contr_head_2", "contr_head_3", "contr_head_4"], help="which type of contr head is used to create the feature vector fead into contrastive loss")
    parser.add_argument("--using_full_decoder", action='store_true', help="If a true a normal encoder is used (encoding to original seg mask size, if false no normal encoder is used")

    # All arguments with default arguments
    parser.add_argument("--coarse_lambda", type=float, default=1.0, help="Coefficient used for the coarse loss in the overall loss")
    parser.add_argument("--index_range", type=list, default=[0,1], help="defines the indices (range from first idx to last idx) that are used for logging/visualizing seg_masks/predictions/feature_maps in a mini batch during validation")
    parser.add_argument("--load_size", type=int, default=256, help="size to which images will get resized")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--device", default='gpu', type=str, help="device to train on")
    parser.add_argument("--num_workers", default=2, type=int, help="num worker for dataloader")
    parser.add_argument("--debug", default=False, type=bool, help="If True, model is run in fast_dev_run (debug mode)")
    parser.add_argument("--overfit_batches", default=0.0, help="Sanity check, takes single batch for training and tries to overfit (Other values can be chosen, check lightning docs")
    parser.add_argument('--load_json', default=None, help='Load settings from file in json format. Command line options override values in file.')
    hparams = parser.parse_args()

    # create a dictionary from the hparams Namespace
    argparse_dict = vars(hparams)

    if hparams.load_json is not None:
        # open the json file with all the arguments needed for training
        with open(hparams.load_json) as json_file:
            json_dict = json.load(json_file)

        # update the arguments with json content
        argparse_dict.update(json_dict)

        # Create a namespace from the argparse_dict
        hparams = Namespace(**argparse_dict)

    main(hparams)