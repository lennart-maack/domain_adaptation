import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.FDA import FDA_first_train, FDA_self_supervised, UNet_baseline

from utils.data import DataModuleSegmentation

def main_FDA(hparams: Namespace):
    
    if hparams.model_type == "FDA":
        model = FDA_first_train(LB=hparams.LB)
        dm = DataModuleSegmentation(path_to_train_source=hparams.source_data_path, path_to_train_target=hparams.target_data_path,
            path_to_test=hparams.test_data_path, domain_adaptation=True, batch_size=hparams.batch_size, load_size=hparams.load_size)
    
    elif hparams.model_type == "FDA_self_supervised":
        model = FDA_self_supervised(LB=hparams.LB)
        dm = DataModuleSegmentation(path_to_train_source=hparams.source_data_path, path_to_train_target=hparams.target_data_path,
            path_to_test=hparams.test_data_path, domain_adaptation=True, pseudo_labels=True, batch_size=hparams.batch_size, load_size=hparams.load_size)
    
    elif hparams.model_type == "UNet_baseline":
        model = UNet_baseline()
        dm = DataModuleSegmentation(path_to_train_source=hparams.source_data_path, path_to_train_target=hparams.target_data_path,
            path_to_test=hparams.test_data_path, domain_adaptation=True, batch_size=hparams.batch_size, load_size=hparams.load_size)
    else:
        print("No correct model was chosen")
        raise SystemExit()

    checkpoint_callback = ModelCheckpoint(save_top_k=2, dirpath=hparams.checkpoint_dir, monitor="Overall Loss", mode="min") # saves top-K checkpoint based on metric defined with monitor

    wandb_logger = WandbLogger(name=hparams.run_name, project=hparams.project_name, log_model="True")
    
    trainer = pl.Trainer(callbacks=checkpoint_callback, accelerator='gpu', devices=1, logger=wandb_logger, max_epochs=hparams.max_epochs)

    wandb_logger.watch(model)

    trainer.fit(model, dm)

    if hparams.test_data_path is not None:
        trainer.test(ckpt_path="best", datamodule=dm)

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--model_type", type=str, choices=["FDA", "FDA_self_supervised", "UNet_baseline"], help="Name of the model which is used for training, FDA: Fourier Domain Adaptation")
    parser.add_argument("--LB", type=float, default= 0.1, help="value (between (0,1)) of beta that examines how much of the amplitude of x^t replaces the amplitude of x^s. Refer to original paper for more information")
    parser.add_argument("--run_name", type=str, help="Name of the run in wandb")
    parser.add_argument("--project_name", type=str, help="Name of the project in wandb")
    parser.add_argument("--checkpoint_dir", type=str, help="path where the checkpoint (model etc.) should be saved")
    parser.add_argument("--source_data_path", type=str, help="path where the source dataset is stored, use this argument when only using one dataset for training (no domain adaptation)")
    parser.add_argument("--target_data_path", type=str, help="path where the target dataset is stored")
    parser.add_argument("--test_data_path", default=None, type=str, help="path where the test target dataset is stored")
    parser.add_argument("--load_size", type=int, help="size to which images will get resized")
    parser.add_argument("--max_epochs", default=150, type=int, help="maximum epochs for training")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    hparams = parser.parse_args()

    main_FDA(hparams)