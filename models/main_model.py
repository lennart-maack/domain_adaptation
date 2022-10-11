import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.resnet.resnet_backbone import ResNetBackbone

from torchmetrics import Dice

class Coarse_Decoder(nn.Module):
    def __init__(self, dim_in, dim_out=1, coarse_decoder="linear") -> None:
        super().__init__()

        if coarse_decoder == "linear":
            self.decoder = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

        elif coarse_decoder == "mlp":
            self.decoder = nn.Sequential(
                nn.Conv2d(dim_in, dim_in/2, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in/2, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        return self.decoder(x)


class MainNetwork(pl.LightningModule):
    def __init__(self, lr=1e-3) -> None:
        super().__init__()
        self.lr = lr
        self.seg_metric = Dice()
        
        self.resnet_backbones = ResNetBackbone()
        self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8()
        self.decoder = Coarse_Decoder(self.encoder.num_features)

    def forward(self, input_data):

        output_features = self.encoder(input_data)

        coarse_prediction = self.decoder(output_features[-1])

        return coarse_prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer
    

    def training_step(self, batch, batch_idx):

        # batch consists of only source images -> batch= img, mask

        img, mask, mask_coarse = batch

        coarse_output = self(img)

        coarse_loss = F.binary_cross_entropy_with_logits(coarse_output, mask_coarse)

        self.log("Training Loss (Binary Cross Entropy)", coarse_loss, prog_bar=True)

        return coarse_loss

    def validation_step(self, batch, batch_idx):

        img, mask, mask_coarse = batch

        coarse_output = self(img)

        val_loss = F.binary_cross_entropy_with_logits(coarse_output, mask_coarse)

        self.seg_metric.update(coarse_output, mask_coarse.to(dtype=torch.uint8))

        return val_loss

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`

        loss = torch.stack(outs).mean()
        dice = self.seg_metric.compute()

        self.log("Validation Loss (Binary Cross Entropy)", loss, prog_bar=True)
        self.log("Dice Score (Validation)", dice, prog_bar=True)

        self.seg_metric.reset()