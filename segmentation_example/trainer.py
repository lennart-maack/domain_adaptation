from segmentation_model import SegModel
from data import DataModuleSegmentation
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from pytorch_lightning.loggers import WandbLogger


def main(hparams: Namespace):
    

    model = SegModel(model_name=hparams.model_name)

    dm = DataModuleSegmentation(hparams.data_path, load_size=hparams.load_size)
    wandb_logger = WandbLogger(project="CVC-ClinicDB Segmentation testing", log_model="False")

    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs)

    wandb_logger.watch(model)

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    parser.add_argument("--load_size", type=int, help="size to which images will get resized")
    parser.add_argument("--model_name", default=None, type=str, help="which segmentation model to use")
    parser.add_argument("--max_epochs", default=100, type=int, help="maximum epochs for training")
    hparams = parser.parse_args()

    main(hparams)