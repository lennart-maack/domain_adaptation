import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.DANN_Segmentation import DANN_Network
from models.segmentation_models import SegModel

from utils.segmentation_metrics import Dice_Coefficient
from utils.data import DataModuleSegmentation

def main(hparams: Namespace):

    if hparams.model_type == "w/o_DA":

        model = SegModel(model_name="unet_resnet_backbone", seg_metric=Dice_Coefficient())
        dm = DataModuleSegmentation(path_to_train_source=hparams.source_data_path, path_to_test=hparams.test_data_path, 
            domain_adaptation=False, batch_size=hparams.batch_size, load_size=hparams.load_size)

        checkpoint_callback = ModelCheckpoint(monitor="Dice Score - Source Data", mode="max")

    elif hparams.model_type == "DANN":

        model = DANN_Network(max_epochs=hparams.max_epochs, batch_size=hparams.batch_size, seg_metric=Dice_Coefficient())
        dm = DataModuleSegmentation(path_to_train_source=hparams.source_data_path, path_to_train_target=hparams.target_data_path,
            path_to_test=hparams.test_data_path, domain_adaptation=True, batch_size=hparams.batch_size, load_size=hparams.load_size)
        
        checkpoint_callback = ModelCheckpoint(monitor="Total Loss - DANN Training", mode="min")

    else:
        print("No allowable model was chosen")
        return

    wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, log_model="True")
    
    trainer = pl.Trainer(callbacks=checkpoint_callback, checkpoint_callback=True, accelerator='gpu', devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs)

    wandb_logger.watch(model)

    trainer.fit(model, dm)

    if hparams.test_data_path is not None:
        trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--model_type", type=str, choices=["w/o_DA", "DANN"], help="Name of the model which is used for training")
    parser.add_argument("--run_name", type=str, help="Name of the run in wandb")
    parser.add_argument("--project_name", type=str, help="Name of the project in wandb")
    parser.add_argument("--source_data_path", type=str, help="path where the source dataset is stored, use this argument when only using one dataset for training (no domain adaptation)")
    parser.add_argument("--target_data_path", type=str, help="path where the target dataset is stored")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--load_size", type=int, help="size to which images will get resized")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    hparams = parser.parse_args()

    main(hparams)