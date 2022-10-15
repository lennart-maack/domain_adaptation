import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.resnet.resnet_backbone import ResNetBackbone

from torchmetrics import Dice

import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torchvision
import torchvision.transforms.functional as F_vision

import wandb

from torchvision import models


def visualize_feature_embedding_matplot(coarse_seg_mask, feature_embedding):
    """
    Visualizes the feaure embedding in a grid. 
    coarse_seg_mask (torch.tensor): (B, NC, H*, W*) B:batch size, NC:number of classes (usually 1), H*,W*: height and width of the segmentation mask
    feature_embedding (torch.tensor): (B, D, H*, W*) B:batch size, D:spatial feature dimension, H*,W*: height and width of the feature embedding
    """

    # we split the feature embedding, because else the computation for visualization would take too long
    splited_embedding_size = int(feature_embedding.size()[1]/2)
    splited_feature_embedding, _ = feature_embedding.split(splited_embedding_size, dim=1)

    spatial_dim = splited_feature_embedding.size()[1]

    grid_dim = math.ceil(math.sqrt(spatial_dim))

    fig = plt.figure(figsize=(grid_dim, grid_dim))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(grid_dim+1, grid_dim),  # creates n x m grid of axes, + 1 because we also plot the segmentation mask in (0,0) 
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    coarse_seg_mask = F_vision.to_pil_image(coarse_seg_mask[0][0]).convert("L")
    coarse_seg_mask = np.asarray(coarse_seg_mask)
    grid[0].imshow(coarse_seg_mask, cmap='gray', vmin=0, vmax=255)

    for i, feature_embed in enumerate(splited_feature_embedding[0]):

        # Iterating over the grid returns the Axes.
        feature_map = F_vision.to_pil_image(feature_embed).convert("L")
        feature_map = np.asarray(feature_map)
        grid[i+1].imshow(feature_map, cmap='gray', vmin=0, vmax=255)

    return fig


def visualize_feature_embedding_torch(feature_embedding, feature_embed_prop, idx):
    """
    Visualizes the feaure embedding in a grid. 
    feature_embedding (torch.tensor): (B, D, H*, W*) B:batch size, D:spatial feature dimension, H*,W*: height and width of the feature embedding
    feature_embed_prop (float): The proportion of feature embeddings that is visualized in the grid. E.g. D=512 and feature_embed_prop=0.5
        -> 256 feature embeddings are visualized in the grid
    idx: (int): Which index of the B mini batches is visualized 
    """

    # we split the feature embedding from size (B, D, H, W) into (B, D/4, H, W), because else the computation for visualization would take too long and lacking clarity
    splited_embedding_size = int(feature_embedding.size()[1] * feature_embed_prop)
    splited_feature_embedding_list = feature_embedding.split(splited_embedding_size, dim=1)

    spatial_dim = splited_feature_embedding_list[0].size()[1]
    grid_dim = math.ceil(math.sqrt(spatial_dim))

    single_feature_embed = splited_feature_embedding_list[idx][0]
    single_feature_embed = single_feature_embed.view(single_feature_embed.size()[0], 1, single_feature_embed.size()[1], single_feature_embed.size()[2])

    single_feature_embed_sigmoid = torch.sigmoid(single_feature_embed)

    grid = torchvision.utils.make_grid(single_feature_embed, nrow=grid_dim)
    grid_sigmoid = torchvision.utils.make_grid(single_feature_embed_sigmoid, nrow=grid_dim)

    return grid, grid_sigmoid


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class Coarse_Decoder(nn.Module):
    def __init__(self, dim_in, dim_out=1, coarse_decoder_type="linear") -> None:
        super().__init__()

        if coarse_decoder_type == "linear":
            self.coarse_decoder = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

        elif coarse_decoder_type == "mlp":
            self.coarse_decoder = nn.Sequential(
                nn.Conv2d(dim_in, 256, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, dim_out, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        return self.coarse_decoder(x)


class Dilated_Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.num_classes = num_classes

        self.bridge_layer0 = convrelu(128, 64, 1, 0)
        self.bridge_layer1 = convrelu(64, 64, 1, 0)
        self.bridge_layer2 = convrelu(128, 128, 1, 0)
        self.bridge_layer3 = convrelu(256, 256, 1, 0)
        self.bridge_layer4 = convrelu(512, 512, 1, 0)

        self.conv_up_layer3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up_layer2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up_layer1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up_layer0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_up_base_layer = convrelu(64 + 128, 64, 3, 1)

        self.upsample_2_to_1 = nn.Upsample(size=(65, 65), mode='bilinear', align_corners=True)
        self.upsample_1_to_0 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.upsample_0_to_original = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

        self.conv_layer_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_layer_original_size1 = convrelu(64, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, self.num_classes, 1)


    def forward(self, input, layer0_output, layer1_output, layer2_output, layer3_output, layer4_output):

        base_layer_output0 = self.conv_layer_original_size0(input)
        base_layer_output1 = self.conv_layer_original_size1(base_layer_output0)

        # layer4 to layer3
        layer4 = self.bridge_layer4(layer4_output)
        layer3 = self.bridge_layer3(layer3_output)
        x = torch.cat([layer4, layer3], dim=1)
        x = self.conv_up_layer3(x)

        # layer3 to layer2
        layer2 = self.bridge_layer2(layer2_output)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up_layer2(x)

        # layer2 to layer1
        x = self.upsample_2_to_1(x)
        layer1 = self.bridge_layer1(layer1_output)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up_layer1(x)

        # layer1 to layer0
        x = self.upsample_1_to_0(x)
        layer0 = self.bridge_layer0(layer0_output)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up_layer0(x)

        # layer0 to original_layer
        x = self.upsample_0_to_original(x)
        x = torch.cat([x, base_layer_output1], dim=1)
        x = self.conv_up_base_layer(x)

        # original_layer to output
        segmentation_mask = self.conv_last(x)

        return segmentation_mask


class Normal_Decoder(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        self.num_classes = num_classes

        self.bridge_layer0 = convrelu(64, 64, 1, 0)
        self.bridge_layer1 = convrelu(64, 64, 1, 0)
        self.bridge_layer2 = convrelu(128, 128, 1, 0)
        self.bridge_layer3 = convrelu(256, 256, 1, 0)
        self.bridge_layer4 = convrelu(512, 512, 1, 0)

        self.conv_up_layer3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up_layer2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up_layer1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up_layer0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_up_base_layer = convrelu(64 + 128, 64, 3, 1)

        self.upsample_4_to_3 = nn.Upsample(size=(17, 17), mode='bilinear', align_corners=True)
        self.upsample_3_to_2 = nn.Upsample(size=(33, 33), mode='bilinear', align_corners=True)
        self.upsample_2_to_1 = nn.Upsample(size=(65, 65), mode='bilinear', align_corners=True)
        self.upsample_1_to_0 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.upsample_0_to_original = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

        self.conv_layer_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_layer_original_size1 = convrelu(64, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, self.num_classes, 1)


    def forward(self, input, layer0_output, layer1_output, layer2_output, layer3_output, layer4_output):

        base_layer_output0 = self.conv_layer_original_size0(input)
        base_layer_output1 = self.conv_layer_original_size1(base_layer_output0)

        # layer4 to layer3
        layer4 = self.bridge_layer4(layer4_output)
        x = self.upsample_4_to_3(layer4)
        layer3 = self.bridge_layer3(layer3_output)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up_layer3(x)

        # layer3 to layer2
        x = self.upsample_3_to_2(x)
        layer2 = self.bridge_layer2(layer2_output)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up_layer2(x)

        # layer2 to layer1
        x = self.upsample_2_to_1(x)
        layer1 = self.bridge_layer1(layer1_output)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up_layer1(x)

        # layer1 to layer0
        x = self.upsample_1_to_0(x)
        layer0 = self.bridge_layer0(layer0_output)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up_layer0(x)

        # layer0 to original_layer
        x = self.upsample_0_to_original(x)
        x = torch.cat([x, base_layer_output1], dim=1)
        x = self.conv_up_base_layer(x)

        # original_layer to output
        segmentation_mask = self.conv_last(x)

        return segmentation_mask



class MainNetwork(pl.LightningModule):
    def __init__(
                self,
                index_range, # defines the indices (range from first idx to last idx) that are used for logging/visualizing seg_masks/predictions/feature_maps in a mini batch during validation
                model_type,
                coarse_prediction_type,
                coarse_lambda,
                lr=1e-3
    ):
        super().__init__()
        
        self.lr = lr
        self.coarse_seg_metric = Dice()
        self.seg_metric = Dice()
        
        self.resnet_backbones = ResNetBackbone()

        if model_type == "dilated":
            print("Using a dilated ResNet18 backbone", "\n")
            self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8()
            self.decoder = Dilated_Decoder()

        elif model_type == "normal":
            print("Using a normal ResNet18 backbone", "\n")
            self.encoder = self.resnet_backbones.resnet18()
            self.decoder = Normal_Decoder()

        # Setting the Coarse Prediction type (What is the architecture of the coarse prediction head)
        if coarse_prediction_type == "linear":
            print("Using a coarse Decoder with linear conv layer", "\n")
            self.coarse_decoder = Coarse_Decoder(self.encoder.num_features, coarse_decoder_type="linear")

        elif coarse_prediction_type == "mlp":
            print("Using a coarse Decoder with mlp layers", "\n")
            self.coarse_decoder = Coarse_Decoder(self.encoder.num_features, coarse_decoder_type="mlp")

        else:
            print("Using no coarse Decoder", "\n")

        self.coarse_prediction_type = coarse_prediction_type
        self.coarse_lambda = coarse_lambda

        self.index_range = list(range(index_range[0],index_range[1]+1))


    def forward(self, input_data):

        output = self.encoder(input_data)

        layer0_output, maxpool_output, layer1_output, layer2_output, layer3_output, layer4_output = output

        if self.coarse_prediction_type == "no_coarse":
            coarse_prediction = None

        else:    
            coarse_prediction = self.coarse_decoder(layer4_output)

        segmentation_mask_prediction = self.decoder(input_data, layer0_output, layer1_output, layer2_output, layer3_output, layer4_output)

        return segmentation_mask_prediction, coarse_prediction, layer4_output


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer
    

    def training_step(self, batch, batch_idx):

        # batch consists of only source images -> batch= img, mask

        img, mask, mask_coarse = batch

        segmentation_mask_prediction, coarse_output, _ = self(img)

        loss = F.binary_cross_entropy_with_logits(segmentation_mask_prediction, mask)

        self.log("Training Loss (Binary Cross Entropy)", loss, prog_bar=True)

        if self.coarse_prediction_type != "no_coarse":
            coarse_loss = F.binary_cross_entropy_with_logits(coarse_output, mask_coarse)
            self.log("Training Coarse Loss (Binary Cross Entropy)", coarse_loss, prog_bar=True)
            loss = loss + self.coarse_lambda * coarse_loss

        return loss

    def validation_step(self, batch, batch_idx):

        img, mask, mask_coarse = batch

        segmentation_mask_prediction, coarse_output, feature_embedding = self(img)

        if batch_idx == 0 and self.current_epoch % 4 == 0:

            for idx in self.index_range:

                # Get the feature embedding (normal and sigmoid) as wandb.Image for logging
                grid_array, grid_array_sigmoid = visualize_feature_embedding_torch(feature_embedding, feature_embed_prop=0.5, idx=idx)
                grid_array = wandb.Image(grid_array)
                grid_array_sigmoid = wandb.Image(grid_array_sigmoid)

                # Get the true Segmentation Mask as wandb.Image for logging
                coarse_seg_mask = wandb.Image(F_vision.to_pil_image(mask_coarse[idx][0]).convert("L"))
                seg_mask = wandb.Image(F_vision.to_pil_image(mask[idx][0]).convert("L"))
                
                # Get the prediction output (normal and sigmoid) as wandb.Image for logging
                # output_image = wandb.Image(F_vision.to_pil_image(segmentation_mask_prediction[idx][0]).convert("L"))
                output_sigmoid = torch.sigmoid(segmentation_mask_prediction)
                output_segmap_sigmoid = wandb.Image(F_vision.to_pil_image(output_sigmoid[idx][0]).convert("L"))

                # Get the true images for logging
                input_image = wandb.Image(F_vision.to_pil_image(img[idx]))


                if self.coarse_prediction_type != "no_coarse":
                    # Get the coarse prediction output (normal and sigmoid) as wandb.Image for logging
                    coarse_output_image = wandb.Image(F_vision.to_pil_image(coarse_output[idx][0]).convert("L"))
                    coarse_output_sigmoid = torch.sigmoid(coarse_output)
                    coarse_output_segmap_sigmoid = wandb.Image(F_vision.to_pil_image(coarse_output_sigmoid[idx][0]).convert("L"))
    
                    wandb.log({
                            # f"Feature Embedding {idx}": grid_array,
                            f"Feature Embedding Sigmoid {idx}": grid_array_sigmoid,
                            f"Coarse True Segmentation Mask {idx}": coarse_seg_mask,
                            f"True Segmentation Mask {idx}": seg_mask,
                            # f"Coarse Prediction Output {idx}": coarse_output_image,
                            f"Coarse Prediction Output Sigmoid {idx}": coarse_output_segmap_sigmoid,
                            # f"Prediction Output {idx}": output_image,
                            f"Prediction Output Sigmoid {idx}": output_segmap_sigmoid,
                            f"Input image {idx}": input_image
                            })
                else:
                    wandb.log({
                            # f"Feature Embedding {idx}": grid_array,
                            f"Feature Embedding Sigmoid {idx}": grid_array_sigmoid,
                            f"Coarse True Segmentation Mask {idx}": coarse_seg_mask,
                            f"True Segmentation Mask {idx}": seg_mask,
                            # f"Prediction Output {idx}": output_image,
                            f"Prediction Output Sigmoid {idx}": output_segmap_sigmoid,
                            f"Input image {idx}": input_image
                            })

        
        val_loss = F.binary_cross_entropy_with_logits(segmentation_mask_prediction, mask)
        self.seg_metric.update(segmentation_mask_prediction, mask.to(dtype=torch.uint8))
        self.log("Validation Loss (BCE)", val_loss, on_step=False, on_epoch=True)

        if self.coarse_prediction_type != "no_coarse":
            coarse_val_loss = F.binary_cross_entropy_with_logits(coarse_output, mask_coarse)
            self.coarse_seg_metric.update(coarse_output, mask_coarse.to(dtype=torch.uint8))
            return {"coarse_vall_loss": coarse_val_loss, "val_loss": val_loss}
        
        else:
            return {"val_loss": val_loss}

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`

        if self.coarse_prediction_type != "no_coarse":
            # coarse_loss = torch.stack([outs[0]["coarse_vall_loss"]]).mean()
            coarse_dice = self.coarse_seg_metric.compute()
            # self.log("Validation Coarse Loss (Binary Cross Entropy)", coarse_loss, prog_bar=True)
            self.log("Coarse Dice Score (Validation)", coarse_dice, prog_bar=True)
            self.coarse_seg_metric.reset()

        # loss = torch.stack([outs[0]["val_loss"]]).mean()
        dice = self.seg_metric.compute()
        # self.log("Validation Loss (Binary Cross Entropy)", loss, prog_bar=True)
        self.log("Dice Score (Validation)", dice, prog_bar=True)
        self.seg_metric.reset()