import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
import json

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.main_model import MainNetwork

from utils.data import DataModuleSegmentation

import wandb


def main(hparams: Namespace, argparse_dict):

    dm = DataModuleSegmentation(path_to_train_source=hparams.path_to_train, load_size=256, coarse_segmentation=33, num_workers=hparams.num_workers)

    checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=hparams.checkpoint_dir, monitor="Dice Score (Validation)", mode="max")

    wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, log_model="True")

    model = MainNetwork()

    trainer = pl.Trainer(callbacks=checkpoint_callback, accelerator=hparams.device, devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs,
                        fast_dev_run=hparams.debug)

    wandb_logger.watch(model)

    wandb.init(config=argparse_dict)

    trainer.fit(model, dm)

    if hparams.test_data_path is not None:
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--run_name", type=str, help="Name of the run in wandb")
    parser.add_argument("--project_name", type=str, help="Name of the project in wandb")
    parser.add_argument("--checkpoint_dir", type=str, help="path where the checkpoint (model etc.) should be saved")
    parser.add_argument("--path_to_train", type=str, help="path where the training data is stored")

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

    main(hparams, argparse_dict)