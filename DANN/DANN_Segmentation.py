from pyexpat import features
import torch
from torch import nn
from functions import ReverseLayerF

import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models

from metrics.segmentation_metrics import Dice_Coefficient

import numpy as np

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

    def forward(self, input):

        base_layer = self.conv_original_size0(input)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        features = self.layer4(layer3)

        return base_layer, layer0, layer1, layer2, layer3, features


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.n_classes = 1
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, base_layer, layer0, layer1, layer2, layer3, features):
        
        # base_layer
        x_original = self.conv_original_size1(base_layer)

        # layer 4 to layer 3
        layer4 = self.layer4_1x1(features)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        # layer 3 to layer 2
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        # layer 2 to layer 1
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        # layer 1 to layer 0
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        # layer 0 to base layer
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        segmentation_mask = self.conv_last(x)

        return segmentation_mask


class Domain_Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, reverse_feature):
        domain_output = self.base(reverse_feature)
        
        return domain_output


class DANN_Network(pl.LightningModule):
    def __init__(self, max_epochs, batch_size):
        super().__init__()
        self.lr = 1e-3
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dice_coeff = Dice_Coefficient()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.domain_classifier = Domain_Classifier()

        self.pooling = nn.AdaptiveAvgPool2d((1,1))

        self.loss_domain = torch.nn.NLLLoss()

    def forward(self, input_data, alpha=2):
        # encoding
        base_layer, layer0, layer1, layer2, layer3, features = self.encoder(input_data)

        # classify domain
        domain_features = self.pooling(features)
        domain_features = domain_features.view(domain_features.size()[0], 512)
        reverse_features = ReverseLayerF.apply(domain_features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        # Get segmentation map
        seg_mask = self.decoder(base_layer, layer0, layer1, layer2, layer3, features)

        return seg_mask, domain_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer

    def training_step(self, batch, batch_idx):

        # batch_idx (int) – Index of current batch.
        batch_source = batch["loader_source"]
        batch_target = batch["loader_target"]

        # self.trainer.num_training_batches is equal to the number of batches (len(torch.data.Dataloader))
        p = float(batch_idx + self.current_epoch * self.trainer.num_training_batches) / self.max_epochs / self.trainer.num_training_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        img_source, mask_source = batch_source
        domain_label_source = torch.zeros(img_source.size()[0]).long().cuda()

        seg_mask_out_source, domain_output_source = self(img_source, alpha)

        err_mask_source = F.binary_cross_entropy_with_logits(seg_mask_out_source, mask_source)
        err_domain_source = self.loss_domain(domain_output_source, domain_label_source)

        # training model using target data
        img_target, _ = batch_target

        domain_label_target = torch.ones(img_target.size()[0]).long().cuda()

        _ , domain_output_target = self(img_target, alpha)

        err_domain_target = self.loss_domain(domain_output_target, domain_label_target)

        loss = err_mask_source + err_domain_source + err_domain_target

        self.log('Error Masks Source - Training', err_mask_source, prog_bar=True)
        self.log('Error Domain Source - Training', err_domain_source, prog_bar=True)
        self.log('Error Domain Target - Training', err_domain_target, prog_bar=True)
        self.log('Loss - Training', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Because in a realistic scenario, there will be no labels for the target dataset,
        we can only validate with the source dataset. Validation will conducted only on 20%
        source data with labels
        """

        img, mask = batch

        seg_out, domain_output = self(img)

        loss_val = F.binary_cross_entropy_with_logits(seg_out, mask)
        dice_coeff_values = self.dice_coeff(seg_out, mask)
        curr_mean_dice = torch.mean(dice_coeff_values[:, -2], dim=0)

        # Log loss, metric and domain output
        self.log('Validation Loss - Source Data', loss_val, prog_bar=True)
        self.log('Dice Score - Source Data', curr_mean_dice, prog_bar=True)
        self.log('Domain Output - Source Data', domain_output, prog_bar=True)

        return {"val_loss": loss_val}