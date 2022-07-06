from DANN_Segmentation import DANN_Network
from data import DataModuleSegmentation
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main(hparams: Namespace):
    

    model = DANN_Network(max_epochs=hparams.max_epochs, batch_size=hparams.batch_size)

    dm = DataModuleSegmentation(hparams.source_data_path, hparams.target_data_path, batch_size=hparams.batch_size, load_size=hparams.load_size)
    wandb_logger = WandbLogger(project="CVC-ClinicDB ETIS DANN", log_model="True")
    checkpoint_callback = ModelCheckpoint(monitor="Loss - Training", mode="min")
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=hparams.max_epochs)

    wandb_logger.watch(model)

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--source_data_path", type=str, help="path where the source dataset is stored")
    parser.add_argument("--target_data_path", type=str, help="path where the target dataset is stored")
    parser.add_argument("--load_size", type=int, help="size to which images will get resized")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    hparams = parser.parse_args()

    main(hparams)