import torch
from torch import nn
from torch.autograd import forward_ad
import torch.nn.functional as F
import PIL
import cv2
import seaborn as sns

import pytorch_lightning as pl

from main_method.utils.visualisation import create_tsne
from main_method.utils.losses import PixelContrastLoss, PixelContrastLoss_Pretrain

from main_method.models.resnet.resnet_backbone import ResNetBackbone

from torchmetrics import Dice 
from torchmetrics.classification import MulticlassJaccardIndex

import math
import numpy as np

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as F_vision

import wandb

import os
import sys
# Enable module loading from parentfolder
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Baseline_and_FDA.utils.FFT import FDA_source_to_target


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



class Dilated_Decoder(nn.Module):
    def __init__(self, num_classes=2):
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

class Contrastive_Head(nn.Module):
    def __init__(
                self,
                dim_in,
                proj_dim=256,
                head_type=None,
    ):
        super().__init__()

        if head_type == "contr_head_1":
            self.contr_head = nn.Sequential(
                    nn.Conv2d(dim_in, dim_in, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_in, proj_dim, kernel_size=1),
                )
        else:
            raise ValueError(head_type, "is not implemented yet")

    def forward(self, feature_embedding):

        x = self.contr_head(feature_embedding)

        return (F.normalize(x, p=2, dim=1))

class Normal_Decoder(nn.Module):

    def __init__(self, num_classes=2):
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


class PreTrain(pl.LightningModule):
    def __init__(
                self,
                wandb_configs
    ):
        super().__init__()

        self.contr_head_type = wandb_configs.contr_head_type

        self.learning_rate = wandb_configs.lr
        self.sup_contr_loss = PixelContrastLoss_Pretrain(temperature=wandb_configs.temperature, base_temperature=wandb_configs.base_temperature, max_samples=wandb_configs.max_samples, max_views=wandb_configs.max_views)
        self.visualize_tsne = wandb_configs.visualize_tsne
        self.use_contr_head_for_tsne = wandb_configs.use_contr_head_for_tsne


        ##########################################################
        # Set the resNet Backbone + full Decoder
        self.resnet_backbones = ResNetBackbone()
        if wandb_configs.model_type == "dilated":
            if wandb_configs.pretrained_ImageNet:
                print("Using a dilated ResNet18 backbone (pretrained)", "\n")
                self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8()
            else:
                print("Using a dilated ResNet18 backbone (NOT pretrained)", "\n")
                self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8_no_pretrain()

        elif wandb_configs.model_type == "normal":
            print("Using a normal ResNet18 backbone", "\n")
            self.encoder = self.resnet_backbones.resnet18()
            
        else:
            raise NotImplementedError("Choose a valid model_type")


        ###########################################################
        # Choose if FDA should be used
        self.apply_FDA = wandb_configs.apply_FDA

        ###########################################################
        # Choosing the contrastive head type to use
        if wandb_configs.contr_head_type == "no_contr_head":
            print("Using No contrastive head/loss")
            self.contr_head = None
        else:
            print("Using contrastive head: ", wandb_configs.contr_head_type)
            self.contr_head = Contrastive_Head(512, head_type=wandb_configs.contr_head_type)
        
        self.contrastive_lambda = wandb_configs.contrastive_lambda

        self.use_pseudo_labels = wandb_configs.use_pseudo_labels
        self.use_confidence_threshold_m_t = wandb_configs.use_confidence_threshold_m_t
        self.use_cycle_gan_source = wandb_configs.use_cycle_gan_source

    def forward(self, input_data):

        output = self.encoder(input_data)

        layer0_output, maxpool_output, layer1_output, layer2_output, layer3_output, layer4_output = output
        
        return layer4_output
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        # return {"optimizer": optimizer}
        
        return {"optimizer": optimizer, "lr_scheduler": { "scheduler": lr_scheduler, "interval": "epoch"} }

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        
        # update params
        optimizer.step(closure=optimizer_closure)
        # skip the first epoch (calculated in steps)
        if self.trainer.global_step < 50:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 30.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

    
    def training_step(self, batch, batch_idx):


        # 0. Load images as batches
        batch_source = batch["loader_source"]
        batch_target = batch["loader_target"]

        img_source, mask_source = batch_source
        
        if self.use_pseudo_labels:
            img_target, pseudo_label_target, m_t = batch_target

        else:
            img_target, _ = batch_target

        # 1. Get pixel embedding from encoder for source img
        layer4_output_src = self(img_source)

        # 2. Calculate FDA transformed source img, get FDA entropy loss, get pixel embedding from encoder for FDA translated source img
        if self.apply_FDA and not self.use_cycle_gan_source:

            src_in_trg = FDA_source_to_target(img_source, img_target, L=0.01)
            loss_ent = self._calculate_FDA_loss_ent(img_target)
            self.log("Training Entropy Loss (Charbonnier)", loss_ent, prog_bar=True)

            layer4_output_src_in_trgt = self(src_in_trg)

        elif self.use_cycle_gan_source and not self.apply_FDA:
            loss_ent = 0
            batch_cycle_gan_source = batch["batch_cycle_gan_source"]
            img_cycle_gan_source, _ = batch_cycle_gan_source
            layer4_output_cycle_gan_source = self(img_cycle_gan_source)

        elif self.use_cycle_gan_source and self.apply_FDA:

            raise AssertionError("Can not use both cycle_gan_source adapted images and FDA at the same time")

        # 2. Get pixel embedding from encoder for target img
        if self.use_pseudo_labels:
            layer4_output_trgt = self(img_target)

        # 3. Calculate Contrastive Loss
        if self.contr_head_type != "no_contr_head":
            
            if self.apply_FDA:
               
                if self.use_pseudo_labels:
                
                    if self.use_confidence_threshold_m_t:
                        contrastive_loss = self._get_contr_loss_pretraining(layer4_output_src=layer4_output_src, mask_source=mask_source, 
                                                                        layer4_output_src_in_trgt=layer4_output_src_in_trgt,
                                                                        layer4_output_trgt=layer4_output_trgt, pseudo_mask_target=pseudo_label_target,
                                                                        m_t=m_t)
                    else:
                        contrastive_loss = self._get_contr_loss_pretraining(layer4_output_src=layer4_output_src, mask_source=mask_source, 
                                                                            layer4_output_src_in_trgt=layer4_output_src_in_trgt,
                                                                            layer4_output_trgt=layer4_output_trgt, pseudo_mask_target=pseudo_label_target)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)

                else:    
                    # Feature embeddings of source and sourceToTarget are to be used
                    contrastive_loss = self._get_contr_loss_pretraining(layer4_output_src=layer4_output_src, mask_source=mask_source, 
                                                                        layer4_output_src_in_trgt=layer4_output_src_in_trgt)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)
            
            elif self.use_cycle_gan_source:
                # Feature embeddings of source and sourceToTarget are to be used
                    contrastive_loss = self._get_contr_loss_pretraining(layer4_output_src=layer4_output_src, mask_source=mask_source, 
                                                                        layer4_output_src_in_trgt=layer4_output_cycle_gan_source)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)
            else:
                
                if self.use_pseudo_labels:
                    contrastive_loss = self._get_contr_loss_pretraining(layer4_output_src=layer4_output_src, mask_source=mask_source,
                                                                        layer4_output_trgt=layer4_output_trgt, pseudo_mask_target=pseudo_label_target)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)

                else:
                    contrastive_loss = self._get_contr_loss_pretraining(layer4_output_src=layer4_output_src, mask_source=mask_source)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)
        else:
            contrastive_loss = 0    
        

        overall_loss = 0.005 * loss_ent + self.contrastive_lambda * contrastive_loss
        self.log("Overall Loss Training", overall_loss, prog_bar=True)

        return overall_loss



    def validation_step(self, batch, batch_idx):

        # Validation is only conducted every few epoch to have faster training!
        if batch_idx == 0:
        
            batch_val = batch["loader_val_source"]
            img_val_src, mask_val_src = batch_val # mask.size(): [bs, 256, 256]
            
            input_image_val_0 = wandb.Image(F_vision.to_pil_image(img_val_src[0]))
            input_image_val_1 = wandb.Image(F_vision.to_pil_image(img_val_src[1]))
            wandb.log({
                    f"input_image_val_source_0": input_image_val_0,
                    f"input_image_val_source_1": input_image_val_1,
                    })

            feature_embedding_val_src = self(img_val_src)

            batch_target_val = batch["loader_target_val"]
            if self.use_pseudo_labels:
                img_target, pseudo_label_target, _ = batch_target_val
                feature_embedding_val_trgt = self(img_target)

            else:
                pseudo_label_target = None
                feature_embedding_val_trgt=None
                img_target, _ = batch_target_val

            if self.apply_FDA:
                img_src_in_trg = FDA_source_to_target(img_val_src, img_target, L=0.01)
                feature_embedding_val_src_in_trgt = self(img_src_in_trg)

                input_image_fda_val_0 = wandb.Image(F_vision.to_pil_image(img_src_in_trg[0]))
                input_image_fda_val_1 = wandb.Image(F_vision.to_pil_image(img_src_in_trg[1]))
                wandb.log({
                        f"input_image_fda_val_0": input_image_fda_val_0,
                        f"input_image_fda_val_1": input_image_fda_val_1,
                        })

            elif self.use_cycle_gan_source:
                batch_cycle_gan_source_val = batch["batch_cycle_gan_source_val"]
                img_cycle_gan_source_val, _ = batch_cycle_gan_source_val

                if batch_idx == 0:
                    input_image_gan_source_val = wandb.Image(F_vision.to_pil_image(img_cycle_gan_source_val[0]))
                    wandb.log({
                            f"input_image_gan_source_val": input_image_gan_source_val,
                            })
                feature_embedding_cycle_gan_source_val = self(img_cycle_gan_source_val)

            if self.visualize_tsne:
                if self.apply_FDA:
                    self._visualize_tsne(feature_embedding_val_src, feature_embedding_val_src_in_trgt, mask_val_src, 
                                        feature_embedding_val_trgt, pseudo_label_target=pseudo_label_target)
                elif self.use_cycle_gan_source:
                    self._visualize_tsne(feature_embedding_val_src, feature_embedding_cycle_gan_source_val, mask_val_src, 
                                        feature_embedding_val_trgt, pseudo_label_target=pseudo_label_target)

        return


    def predict_step(self, batch, batch_idx):
        
        # THIS STEP IS USED TO VISUALIZE TSNE - USE visulize_tsne.py
        if batch_idx == 0:
        
            batch_val = batch["loader_val_source"]
            img_val_src, mask_val_src = batch_val # mask.size(): [bs, 256, 256]
            
            feature_embedding_val_src = self(img_val_src)

            batch_target_val = batch["loader_target_val"]
            if self.use_pseudo_labels:
                img_target, pseudo_label_target, mask_target = batch_target_val
                feature_embedding_val_trgt = self(img_target)

            else:
                pseudo_label_target = None
                img_target, mask_target = batch_target_val
                feature_embedding_val_trgt = self(img_target)

            if self.apply_FDA and not self.use_cycle_gan_source:
                img_src_in_trg = FDA_source_to_target(img_val_src, img_target, L=0.01)
                feature_embedding_val_src_in_trgt = self(img_src_in_trg)

            elif self.use_cycle_gan_source and not self.apply_FDA:
                print("Use cycle GAN for predicting/creating tsne")
                batch_source_to_target_cycle = batch["batch_cycle_gan_source_val"]
                img_val_src_in_trgt_cycle, _ = batch_source_to_target_cycle
                feature_embedding_val_src_in_trgt = self(img_val_src_in_trgt_cycle)

            elif self.use_cycle_gan_source and self.apply_FDA:
                raise AssertionError("Can not use both cycle_gan_source adapted images and FDA at the same time")


            self._visualize_tsne(feature_embedding_val_src, feature_embedding_val_src_in_trgt, mask_val_src, 
                                feature_embedding_val_trgt, mask_val_target=mask_target, perplexity=30)

        return




    def _calculate_FDA_loss_ent(self, image_target):

        # 2. Compute the Entropy Loss from the target image (Needed for Regularization with Charbonnier penalty function)
        ## Is it needed to calculate softmax etc. if it is 1 class only anyway? --> I apply sigmoid here/Have a look on how the out values look like!!
        out = self(image_target)[0] # shape of out is: B, 1 (N_Classes), 256 (H), 256 (W)
        P = F.softmax(out, dim=1)        # [B, 19, H, W] (from paper), in our case: [B, 1, H, W] 
        # P = F.sigmoid(out)
        logP = F.log_softmax(out, dim=1) # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        # logP = F.logsigmoid(out)
        PlogP = P * logP               # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W] (from paper)
        # ent = ent / 2.9444         # chanage when classes is not 19 (from paper), we leave out this line?? 
        
        # compute robust entropy/Charbonnier penalty function
        ent = ent ** 2.0 + 1e-8
        eta = 2.0
        ent = ent ** eta
        loss_ent = ent.mean()

        return loss_ent


    def _get_contr_loss_pretraining(self, layer4_output_src, mask_source, layer4_output_src_in_trgt=None, 
                                    layer4_output_trgt = None, pseudo_mask_target = None, m_t = None):

        if layer4_output_src_in_trgt is not None:
            
            if layer4_output_trgt is not None:
                feature_embed_list = [layer4_output_src, layer4_output_src_in_trgt, layer4_output_trgt]
                mask_list = [mask_source, mask_source, pseudo_mask_target]
            else:
                feature_embed_list = [layer4_output_src, layer4_output_src_in_trgt]
                mask_list = [mask_source, mask_source]

        else:
            if layer4_output_trgt is not None:
                feature_embed_list = [layer4_output_src, layer4_output_trgt]
                mask_list = [mask_source, pseudo_mask_target]
            else:
                feature_embed_list = [layer4_output_src]
                mask_list = [mask_source]

            
        concat_feature_embed = torch.cat(feature_embed_list, dim=0)
        concat_masks = torch.cat(mask_list, dim=0)
        concat_contrastive_embedding = self.contr_head(concat_feature_embed)
        
        labels = concat_masks
        
        if layer4_output_trgt is not None and m_t is not None:

            m_t_ones_length = concat_masks.size(0) - pseudo_mask_target.size(0)
            m_t_ones = torch.ones(m_t_ones_length, concat_masks.size(1), concat_masks.size(2)).to(device=self.device)
            
            m_t_concat = torch.cat([m_t_ones, m_t])

            contrastive_loss = self.sup_contr_loss(concat_contrastive_embedding, labels, m_t=m_t_concat)

        else:
            contrastive_loss = self.sup_contr_loss(concat_contrastive_embedding, labels)

        return contrastive_loss


    def _visualize_tsne(self, feature_embedding_val_src, feature_embedding_val_src_in_trgt, mask_val_src, 
                                feature_embedding_val_trgt=None, mask_val_target=None,
                                pca_n_comp=50, perplexity=30):

        mask_val_src = mask_val_src.unsqueeze(1).float().clone()
        mask_val_src = torch.nn.functional.interpolate(mask_val_src,
                                                    (feature_embedding_val_src.shape[2], feature_embedding_val_src.shape[3]), mode='nearest')
        mask_val_src = mask_val_src.squeeze(1).long()

        if mask_val_target is not None:
            
            if self.use_contr_head_for_tsne:
                feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt = self._get_contr_feat_embeds(feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt)

            mask_val_target = mask_val_target.unsqueeze(1).float().clone()
            mask_val_target = torch.nn.functional.interpolate(mask_val_target,
                                                    (feature_embedding_val_src.shape[2], feature_embedding_val_src.shape[3]), mode='nearest')
            mask_val_target = mask_val_target.squeeze(1).long()
            concat_feature_embeds = torch.tensor([feature_embedding_val_src.cpu().numpy(), feature_embedding_val_src_in_trgt.cpu().numpy(), 
                                                feature_embedding_val_trgt.cpu().numpy()])
            concat_labels = torch.tensor([mask_val_src.cpu().numpy(), mask_val_src.cpu().numpy(), mask_val_target.cpu().numpy()])
            df_tsne = create_tsne(concat_feature_embeds, concat_labels, pca_n_comp=pca_n_comp, perplexity=perplexity)

            df_tsne = self._clean_up_tsne(df_tsne, with_target_data=True)


        else:
            if self.use_contr_head_for_tsne:
                feature_embedding_val_src, feature_embedding_val_src_in_trgt= self._get_contr_feat_embeds(feature_embedding_val_src,
                                                                                                                        feature_embedding_val_src_in_trgt)

            concat_feature_embeds = torch.tensor([feature_embedding_val_src.cpu().numpy(), feature_embedding_val_src_in_trgt.cpu().numpy()])
            concat_labels = torch.tensor([mask_val_src.cpu().numpy(), mask_val_src.cpu().numpy()])
            df_tsne = create_tsne(concat_feature_embeds, concat_labels, pca_n_comp=pca_n_comp, perplexity=perplexity)
            
            df_tsne = self._clean_up_tsne(df_tsne, with_target_data=False)


        fig = plt.figure(figsize=(10,10))
        sns_plot = sns.scatterplot(x="tsne-one", y="tsne-two", hue="Domain", style="Background", palette="deep", data=df_tsne)
        fig = sns_plot.get_figure()
        fig.savefig(f"tSNE_vis_perplex_{perplexity}.jpg")
        tsne_img = cv2.imread(f"tSNE_vis_perplex_{perplexity}.jpg")
        tSNE_image = wandb.Image(PIL.Image.fromarray(tsne_img))

        wandb.log({f"tSNE visualization perplex {perplexity}": tSNE_image})

    def _get_contr_feat_embeds(self, feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt=None):

        if feature_embedding_val_trgt is not None:

            concat_feats = torch.concat([feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt], dim=0)
            concat_contrastive_embedding = self.contr_head(concat_feats)

            length_concat_contrastive_embedding = concat_contrastive_embedding.size(0)
            a_end_id = int(length_concat_contrastive_embedding/3)
            b_end_id = int(length_concat_contrastive_embedding/3 * 2)
            c_end_id = int(length_concat_contrastive_embedding/3 * 3)
            contr_feature_embedding_val_src = concat_contrastive_embedding[:a_end_id]
            contr_feature_embedding_val_src_in_trgt = concat_contrastive_embedding[a_end_id:b_end_id]
            contr_feature_embedding_val_trgt = concat_contrastive_embedding[b_end_id:c_end_id]

            return contr_feature_embedding_val_src, contr_feature_embedding_val_src_in_trgt, contr_feature_embedding_val_trgt
        else:

            concat_feats = torch.concat([feature_embedding_val_src, feature_embedding_val_src_in_trgt], dim=0)
            concat_contrastive_embedding = self.contr_head(concat_feats)

            length_concat_contrastive_embedding = concat_contrastive_embedding.size(0)
            a_end_id = int(length_concat_contrastive_embedding/2)
            b_end_id = int(length_concat_contrastive_embedding/2 * 2)
            contr_feature_embedding_val_src = concat_contrastive_embedding[:a_end_id]
            contr_feature_embedding_val_src_in_trgt = concat_contrastive_embedding[a_end_id:b_end_id]

            return contr_feature_embedding_val_src, contr_feature_embedding_val_src_in_trgt

    def _clean_up_tsne(self, df_tsne, with_target_data=False):

        if with_target_data:

            df_tsne.loc[((df_tsne['label'] == 0) | (df_tsne['label'] == 1), 'Domain')] = "Source"
            df_tsne.loc[((df_tsne['label'] == 2) | (df_tsne['label'] == 3), 'Domain')] = "Source To Target"
            df_tsne.loc[((df_tsne['label'] == 4) | (df_tsne['label'] == 5), 'Domain')] = "Target"
            df_tsne["Background"] = np.where((df_tsne['label'] == 0) | (df_tsne['label'] == 2) | (df_tsne['label'] == 4) , "Background", "Polyp")

            df_new = df_tsne
            print(df_new['label'].value_counts())
            df_back = df_new.loc[(df_new['label'] == 0) | (df_new['label'] == 2) | (df_new['label'] == 4)]

            np.random.seed(10)
            
            num_polyps = len(df_new.loc[(df_new['label'] == 1)].index) + len(df_new.loc[(df_new['label'] == 1)].index) + len(df_new.loc[(df_new['label'] == 5)].index)
            num_backs = len(df_new.loc[(df_new['label'] == 0)].index) + len(df_new.loc[(df_new['label'] == 0)].index) + len(df_new.loc[(df_new['label'] == 4)].index)
            remove_n = num_backs - num_polyps - 500

            drop_indices_back = np.random.choice(df_back.index, remove_n, replace=False)
            df_minimized_backs = df_new.drop(drop_indices_back)

            ### Minimize amount of polyp sourceToTarget
            num_src_to_trgt_to_cut = int(len(df_new.loc[(df_new['label'] == 3)]) * 0.66)
            drop_indices_src_to_trgt_polyp = np.random.choice(df_new.loc[(df_new['label'] == 3)].index, num_src_to_trgt_to_cut, replace=False)
            df_minimized_backs = df_minimized_backs.drop(drop_indices_src_to_trgt_polyp)
            ###

            print(df_minimized_backs['label'].value_counts())

            return df_minimized_backs

        else:

            df_tsne.loc[((df_tsne['label'] == 0) | (df_tsne['label'] == 1), 'Domain')] = "Source"
            df_tsne.loc[((df_tsne['label'] == 2) | (df_tsne['label'] == 3), 'Domain')] = "Source To Target"
            df_tsne["Background"] = np.where((df_tsne['label'] == 0) | (df_tsne['label'] == 2) , "Background", "Polyp")

            df_tsne['label'] = df_tsne['label'].replace(0,'src - back')
            df_tsne['label'] = df_tsne['label'].replace(1,'src - polyp')
            df_tsne['label'] = df_tsne['label'].replace(2,'src to trgt - back')
            df_tsne['label'] = df_tsne['label'].replace(3,'src to trgt - polyp')
            
            df_new = df_tsne
            print(df_new['label'].value_counts())
            df_back = df_new.loc[(df_new['label'] == "src - back") | (df_new['label'] == "src to trgt - back")]

            np.random.seed(10)
            
            num_polyps = len(df_new.loc[(df_new['label'] == "src - polyp")].index) + len(df_new.loc[(df_new['label'] == "src - polyp")].index)
            num_backs = len(df_new.loc[(df_new['label'] == "src - back")].index) + len(df_new.loc[(df_new['label'] == "src - back")].index)
            remove_n = num_backs - num_polyps - 500

            drop_indices_back = np.random.choice(df_back.index, remove_n, replace=False)
            df_minimized_backs = df_new.drop(drop_indices_back)

            ### Minimize amount of polyp sourceToTarget
            num_src_to_trgt_to_cut = int(len(df_new.loc[(df_new['label'] == "src to trgt - polyp")]) * 0.66)
            drop_indices_src_to_trgt_polyp = np.random.choice(df_new.loc[(df_new['label'] == "src to trgt - polyp")].index, num_src_to_trgt_to_cut, replace=False)
            df_minimized_backs = df_minimized_backs.drop(drop_indices_src_to_trgt_polyp)
            ###


            print(df_minimized_backs['label'].value_counts())

            return df_minimized_backs



class FineTune(pl.LightningModule):
    def __init__(
                self,
                wandb_configs,
                encoder=None,
                contr_head=None
    ):
        super().__init__()

        self.learning_rate = wandb_configs.lr
        self.seg_metric = Dice(num_classes=2, average="macro")
        self.seg_metric_test = Dice(num_classes=2, average="macro")
        self.seg_metric_test_during_val = Dice(num_classes=2, average="macro")
        self.seg_metric_IoU = MulticlassJaccardIndex(num_classes=2)
        self.seg_metric_test_IoU = MulticlassJaccardIndex(num_classes=2)
        self.seg_metric_test_during_val_IoU = MulticlassJaccardIndex(num_classes=2)


        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.visualize_tsne = wandb_configs.visualize_tsne
        self.use_contr_head_for_tsne = wandb_configs.use_contr_head_for_tsne
        # self.contr_head = contr_head
        self.contr_head = None

        ####
        # Set the pretrained encoder and decoder
        if wandb_configs.model_type == "dilated":
            if encoder is not None:
                self.encoder = encoder
            else:
                self.resnet_backbones = ResNetBackbone()
                if wandb_configs.pretrained_ImageNet:
                    print("Using a dilated ResNet18 backbone (pretrained)", "\n")
                    self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8()
                
                else:
                    print("Using a dilated ResNet18 backbone (NOT pretrained)", "\n")
                    self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8_no_pretrain()
            self.decoder = Dilated_Decoder()

        elif wandb_configs.model_type == "normal":
            self.resnet_backbones = ResNetBackbone()
            if wandb_configs.pretrained_ImageNet: 
                print("Using a normal ResNet18 backbone (pretrained)", "\n")
                self.encoder = self.resnet_backbones.resnet18()

            else:
                print("Using a normal ResNet18 backbone (NOT pretrained)", "\n")
                self.encoder = self.resnet_backbones.resnet18_no_pretrained()

            self.decoder = Normal_Decoder()
            
        else:
            raise NotImplementedError("Choose a valid model_type")


        ###########################################################
        # Choose if FDA should be used
        self.apply_FDA = wandb_configs.apply_FDA


        ###########################################################
        # Index range which is used for the amount of images logged
        self.index_range = list(range(wandb_configs.index_range[0],wandb_configs.index_range[1]+1))

        self.use_pseudo_labels = wandb_configs.use_pseudo_labels
        self.use_cycle_gan_source = wandb_configs.use_cycle_gan_source

    def forward(self, input_data):

        output = self.encoder(input_data)

        layer0_output, maxpool_output, layer1_output, layer2_output, layer3_output, layer4_output = output

        segmentation_mask_prediction = self.decoder(input_data, layer0_output, layer1_output, layer2_output, layer3_output, layer4_output)
        
        return segmentation_mask_prediction, layer4_output
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        
        return {"optimizer": optimizer, "lr_scheduler": { "scheduler": lr_scheduler, "interval": "epoch"} }
    

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        
        # update params
        optimizer.step(closure=optimizer_closure)
        # skip the first epoch (calculated in steps)
        if self.trainer.global_step < 30:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 30.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate


    def training_step(self, batch, batch_idx):


        # 0. Load images as batches
        batch_source = batch["loader_source"]
        batch_target = batch["loader_target"]

        img_source, mask_source = batch_source
        
        if self.use_pseudo_labels:
            img_target, pseudo_label_target, m_t = batch_target

        else:
            img_target, _ = batch_target
        

        # 1. Calculate CE loss - source img 
        segmentation_mask_prediction_src, layer4_output_src = self(img_source)
        loss_ce_src = self.cross_entropy_loss(segmentation_mask_prediction_src, mask_source)
        self.log("Training Loss - Source (Cross Entropy)", loss_ce_src, prog_bar=True)

        # 1.2 Calculate CE loss - target img with pseudo labels
        if self.use_pseudo_labels:
            segmentation_mask_prediction_trgt, layer4_output_trgt = self(img_target)
            loss_ce_trgt = self.cross_entropy_loss(segmentation_mask_prediction_trgt, pseudo_label_target)
            self.log("Training Loss - Target Pseudo (Cross Entropy)", loss_ce_trgt, prog_bar=True)
        else:
            loss_ce_trgt = 0

        # 2. Calculate FDA transformed source img and FDA entropy loss and CE loss - target img with pseudo labels
        if self.apply_FDA and not self.use_cycle_gan_source:

            src_in_trg = FDA_source_to_target(img_source, img_target, L=0.01)
            
            loss_ent = self._calculate_FDA_loss_ent(img_target)
            self.log("Training Entropy Loss (Charbonnier)", loss_ent, prog_bar=True)

            segmentation_mask_prediction_src_in_trgt, layer4_output_src_in_trgt = self(src_in_trg)
            loss_ce_src_in_trgt = self.cross_entropy_loss(segmentation_mask_prediction_src_in_trgt, mask_source)
            self.log("Training Loss - Source in Target (Cross Entropy)", loss_ce_src_in_trgt, prog_bar=True)
        
        elif self.use_cycle_gan_source and not self.apply_FDA:
            loss_ent = 0
            batch_cycle_gan_source = batch["batch_cycle_gan_source"]
            img_cycle_gan_source, _ = batch_cycle_gan_source
            segmentation_mask_prediction_src_in_trgt, layer4_output_src_in_trgt = self(img_cycle_gan_source)
            loss_ce_src_in_trgt = self.cross_entropy_loss(segmentation_mask_prediction_src_in_trgt, mask_source)
            self.log("Training Loss - Source in Target with CycleGAN (Cross Entropy)", loss_ce_src_in_trgt, prog_bar=True)

        elif self.use_cycle_gan_source and self.apply_FDA:

            raise AssertionError("Can not use both cycle_gan_source adapted images and FDA at the same time")

        else:
            loss_ent = 0
            loss_ce_src_in_trgt = 0
        
        num_non_zero_losses = len([i for i, e in enumerate([loss_ce_src, loss_ce_src_in_trgt, loss_ce_trgt, loss_ent]) if e != 0])

        overall_loss = (loss_ce_src + loss_ce_src_in_trgt + loss_ce_trgt + 0.01 * loss_ent)/num_non_zero_losses
        self.log("Overall Loss Training", overall_loss, prog_bar=True)

        return overall_loss



    def validation_step(self, batch, batch_idx):

        # 1. Calculate CE Loss and Dice Score on source validation data
        batch_val = batch["loader_val_source"]
        img_val_src, mask_val_src = batch_val # mask.size(): [bs, 256, 256]
        
        segmentation_mask_prediction_val_src, feature_embedding_val_src = self(img_val_src)
        val_loss = self.cross_entropy_loss(segmentation_mask_prediction_val_src, mask_val_src)
        self.seg_metric.update(segmentation_mask_prediction_val_src, mask_val_src.to(dtype=torch.uint8))
        self.seg_metric_IoU.update(segmentation_mask_prediction_val_src, mask_val_src.to(dtype=torch.uint8))
        self.log("Validation CE Loss", val_loss, on_step=False, on_epoch=True)


        # 2. Create pixel_embeddings (feature_embedding) for src_to_trgt (FDA) and trgt_val
        batch_target_val = batch["loader_target_val"]
        if self.use_pseudo_labels:
            img_target, pseudo_label_target, _ = batch_target_val
            _, feature_embedding_val_trgt = self(img_target)

        else:
            feature_embedding_val_trgt = None
            pseudo_label_target = None
            img_target, _ = batch_target_val
        
        if self.apply_FDA and not self.use_cycle_gan_source:
            # 2a. Create pixel_embeddings for src_to_trgt
            img_val_src_in_trg = FDA_source_to_target(img_val_src, img_target, L=0.01)
            _, feature_embedding_val_src_in_trgt = self(img_val_src_in_trg)
        
        if self.use_cycle_gan_source and not self.apply_FDA:
            
            batch_cycle_gan_source_val = batch["batch_cycle_gan_source_val"]
            img_cycle_gan_source_val, _ = batch_cycle_gan_source_val
            _, feature_embedding_val_src_in_trgt = self(img_cycle_gan_source_val)
        
        elif self.use_cycle_gan_source and self.apply_FDA:
            raise AssertionError("Can not use both cycle_gan_source adapted images and FDA at the same time")

        else:
            feature_embedding_val_src_in_trgt = None

        # 3. Calculate Dice Score on target test data
        batch_test = batch["loader_target_test"]
        img_test, mask_test = batch_test

        segmentation_mask_prediction_test, _ = self(img_test)
        self.seg_metric_test_during_val.update(segmentation_mask_prediction_test, mask_test.to(dtype=torch.uint8))
        self.seg_metric_test_during_val_IoU.update(segmentation_mask_prediction_test, mask_test.to(dtype=torch.uint8))

        # 3. Section for visualization
        if batch_idx == 0 and self.current_epoch % 80 == 0:
            
            if self.visualize_tsne and self.apply_FDA:
                print("Calculating tsne")
                self._visualize_tsne(feature_embedding_val_src, feature_embedding_val_src_in_trgt, mask_val_src, 
                                    feature_embedding_val_trgt, pseudo_label_target,
                                    pca_n_comp=50, perplexity=30)

            self._visualize_plots(mask_val_src, segmentation_mask_prediction_val_src, img_val_src)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        
        dice = self.seg_metric.compute()
        self.log("Dice Score (Validation)", dice, prog_bar=True)
        self.seg_metric.reset()
        ioU = self.seg_metric_IoU.compute()
        self.log("IoU (Validation)", ioU, prog_bar=True)
        self.seg_metric_IoU.reset()

        dice_test_during_val = self.seg_metric_test_during_val.compute()
        self.log("Dice Score on Test/Target (Validation)", dice_test_during_val, prog_bar=True)
        self.seg_metric_test_during_val.reset()
        ioU_test_during_val = self.seg_metric_test_during_val_IoU.compute()
        self.log("IoU Score on Test/Target (Validation)", ioU_test_during_val, prog_bar=True)
        self.seg_metric_test_during_val_IoU.reset()

    def test_step(self, batch, batch_idx):

        img, mask = batch # mask.size(): [bs, 1, 256, 256]

        segmentation_mask_prediction, feature_embedding = self(img)
        self.seg_metric_test.update(segmentation_mask_prediction, mask.to(dtype=torch.uint8))
        self.seg_metric_test_IoU.update(segmentation_mask_prediction, mask.to(dtype=torch.uint8))       


    def test_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`

        seg_metric_test = self.seg_metric_test.compute()
        self.log("FINAL Dice Score on Target Test Data", seg_metric_test, prog_bar=True)
        self.seg_metric_test.reset()
        ioU_metric_test = self.seg_metric_test_IoU.compute()
        self.log("FINAL IoU Score on Target Test Data", ioU_metric_test, prog_bar=True)
        self.seg_metric_test_IoU.reset()


    def predict_step(self, batch, batch_idx):
        
        img, mask = batch
        segmentation_mask_prediction_out, _ = self(img)

        segmentation_mask_prediction = torch.argmax(segmentation_mask_prediction_out, 1)
        seg_pred_softmax = torch.nn.Softmax2d()(segmentation_mask_prediction_out)
        max_pred, max_indices = torch.max(seg_pred_softmax, 1)
        m_t = (max_pred > 0.9).long()

        return {"segmentation_mask_prediction": segmentation_mask_prediction, "m_t": m_t}


    def _calculate_FDA_loss_ent(self, image_target):

        # 2. Compute the Entropy Loss from the target image (Needed for Regularization with Charbonnier penalty function)
        ## Is it needed to calculate softmax etc. if it is 1 class only anyway? --> I apply sigmoid here/Have a look on how the out values look like!!
        out = self(image_target)[0] # shape of out is: B, 1 (N_Classes), 256 (H), 256 (W)
        P = F.softmax(out, dim=1)        # [B, 19, H, W] (from paper), in our case: [B, 1, H, W] 
        # P = F.sigmoid(out)
        logP = F.log_softmax(out, dim=1) # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        # logP = F.logsigmoid(out)
        PlogP = P * logP               # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W] (from paper)
        # ent = ent / 2.9444         # chanage when classes is not 19 (from paper), we leave out this line?? 
        
        # compute robust entropy/Charbonnier penalty function
        ent = ent ** 2.0 + 1e-8
        eta = 2.0
        ent = ent ** eta
        loss_ent = ent.mean()

        return loss_ent


    def _get_contr_loss_pretraining(self, layer4_output_src, mask_source, layer4_output_src_in_trgt=None, 
                                    layer4_output_trgt = None, pseudo_mask_target = None, m_t = None):

        contrastive_embedding_src = self.contr_head(layer4_output_src)

        if layer4_output_src_in_trgt is not None:
           
            feature_embed_list = [layer4_output_src, layer4_output_src_in_trgt]
            concat_feature_embed = torch.cat(feature_embed_list, dim=0)
            ## Concat the two corresponding masks
            mask_list = [mask_source, mask_source]
            concat_masks = torch.cat(mask_list, dim=0)
            # downscale the prediction or use the coarse prediction could be implemented - until now we use the coarse prediction and the coarse mask
            concat_contrastive_embedding = self.contr_head(concat_feature_embed)
            labels = concat_masks
        
        else:
            concat_contrastive_embedding = layer4_output_src
            labels = mask_source
        
        # _, labels = torch.max(labels, 1)
        
        contrastive_loss = self.sup_contr_loss(concat_contrastive_embedding, labels)

        return contrastive_loss

    
    def _visualize_tsne(self, feature_embedding_val_src, feature_embedding_val_src_in_trgt, mask_val_src, 
                            feature_embedding_val_trgt=None, pseudo_label_target=None,
                            pca_n_comp=50, perplexity=30):

        mask_val_src = mask_val_src.unsqueeze(1).float().clone()
        mask_val_src = torch.nn.functional.interpolate(mask_val_src,
                                                    (feature_embedding_val_src.shape[2], feature_embedding_val_src.shape[3]), mode='nearest')
        mask_val_src = mask_val_src.squeeze(1).long()

        if pseudo_label_target is not None:
            
            if self.use_contr_head_for_tsne:
                feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt = self._get_contr_feat_embeds(feature_embedding_val_src, 
                                                                                                                                                feature_embedding_val_src_in_trgt, 
                                                                                                                                                feature_embedding_val_trgt)

            pseudo_label_target = pseudo_label_target.unsqueeze(1).float().clone()
            pseudo_label_target = torch.nn.functional.interpolate(pseudo_label_target,
                                                    (feature_embedding_val_src.shape[2], feature_embedding_val_src.shape[3]), mode='nearest')
            pseudo_label_target = pseudo_label_target.squeeze(1).long()
            concat_feature_embeds = torch.tensor([feature_embedding_val_src.cpu().numpy(), feature_embedding_val_src_in_trgt.cpu().numpy(), 
                                                feature_embedding_val_trgt.cpu().numpy()])
            concat_labels = torch.tensor([mask_val_src.cpu().numpy(), mask_val_src.cpu().numpy(), pseudo_label_target.cpu().numpy()])
            df_tsne = create_tsne(concat_feature_embeds, concat_labels, pca_n_comp=pca_n_comp, perplexity=perplexity)

            df_tsne = self._clean_up_tsne(df_tsne, with_target_data=True)


        else:
            if self.use_contr_head_for_tsne:
                feature_embedding_val_src, feature_embedding_val_src_in_trgt= self._get_contr_feat_embeds(feature_embedding_val_src,
                                                                                                                        feature_embedding_val_src_in_trgt)

            concat_feature_embeds = torch.tensor([feature_embedding_val_src.cpu().numpy(), feature_embedding_val_src_in_trgt.cpu().numpy()])
            concat_labels = torch.tensor([mask_val_src.cpu().numpy(), mask_val_src.cpu().numpy()])
            df_tsne = create_tsne(concat_feature_embeds, concat_labels, pca_n_comp=pca_n_comp, perplexity=perplexity)
            
            df_tsne = self._clean_up_tsne(df_tsne, with_target_data=False)


        fig = plt.figure(figsize=(10,10))
        sns_plot = sns.scatterplot(x="tsne-one", y="tsne-two", hue="Domain", style="Background", palette="deep", data=df_tsne)
        fig = sns_plot.get_figure()
        fig.savefig(f"tSNE_vis_perplex_{perplexity}.jpg")
        tsne_img = cv2.imread(f"tSNE_vis_perplex_{perplexity}.jpg")
        tSNE_image = wandb.Image(PIL.Image.fromarray(tsne_img))

        wandb.log({f"tSNE visualization perplex {perplexity}": tSNE_image})


    def _clean_up_tsne(self, df_tsne, with_target_data=False):

        if with_target_data:

            df_tsne.loc[((df_tsne['label'] == 0) | (df_tsne['label'] == 1), 'Domain')] = "Source"
            df_tsne.loc[((df_tsne['label'] == 2) | (df_tsne['label'] == 3), 'Domain')] = "Source To Target"
            df_tsne.loc[((df_tsne['label'] == 4) | (df_tsne['label'] == 5), 'Domain')] = "Target"
            df_tsne["Background"] = np.where((df_tsne['label'] == 0) | (df_tsne['label'] == 2) | (df_tsne['label'] == 4) , "Background", "Polyp")

            df_new = df_tsne
            print(df_new['label'].value_counts())
            df_back = df_new.loc[(df_new['label'] == 0) | (df_new['label'] == 2) | (df_new['label'] == 4)]

            np.random.seed(10)
            
            num_polyps = len(df_new.loc[(df_new['label'] == 1)].index) + len(df_new.loc[(df_new['label'] == 1)].index) + len(df_new.loc[(df_new['label'] == 5)].index)
            num_backs = len(df_new.loc[(df_new['label'] == 0)].index) + len(df_new.loc[(df_new['label'] == 0)].index) + len(df_new.loc[(df_new['label'] == 4)].index)
            remove_n = num_backs - num_polyps - 500

            drop_indices_back = np.random.choice(df_back.index, remove_n, replace=False)
            df_minimized_backs = df_new.drop(drop_indices_back)

            ### Minimize amount of polyp sourceToTarget
            num_src_to_trgt_to_cut = int(len(df_new.loc[(df_new['label'] == 3)]) * 0.66)
            drop_indices_src_to_trgt_polyp = np.random.choice(df_new.loc[(df_new['label'] == 3)].index, num_src_to_trgt_to_cut, replace=False)
            df_minimized_backs = df_minimized_backs.drop(drop_indices_src_to_trgt_polyp)
            ###

            print(df_minimized_backs['label'].value_counts())

            return df_minimized_backs

        else:

            df_tsne.loc[((df_tsne['label'] == 0) | (df_tsne['label'] == 1), 'Domain')] = "Source"
            df_tsne.loc[((df_tsne['label'] == 2) | (df_tsne['label'] == 3), 'Domain')] = "Source To Target"
            df_tsne["Background"] = np.where((df_tsne['label'] == 0) | (df_tsne['label'] == 2) , "Background", "Polyp")

            df_tsne['label'] = df_tsne['label'].replace(0,'src - back')
            df_tsne['label'] = df_tsne['label'].replace(1,'src - polyp')
            df_tsne['label'] = df_tsne['label'].replace(2,'src to trgt - back')
            df_tsne['label'] = df_tsne['label'].replace(3,'src to trgt - polyp')
            
            df_new = df_tsne
            print(df_new['label'].value_counts())
            df_back = df_new.loc[(df_new['label'] == "src - back") | (df_new['label'] == "src to trgt - back")]

            np.random.seed(10)
            
            num_polyps = len(df_new.loc[(df_new['label'] == "src - polyp")].index) + len(df_new.loc[(df_new['label'] == "src - polyp")].index)
            num_backs = len(df_new.loc[(df_new['label'] == "src - back")].index) + len(df_new.loc[(df_new['label'] == "src - back")].index)
            remove_n = num_backs - num_polyps - 500

            drop_indices_back = np.random.choice(df_back.index, remove_n, replace=False)
            df_minimized_backs = df_new.drop(drop_indices_back)

            ### Minimize amount of polyp sourceToTarget
            num_src_to_trgt_to_cut = int(len(df_new.loc[(df_new['label'] == "src to trgt - polyp")]) * 0.66)
            drop_indices_src_to_trgt_polyp = np.random.choice(df_new.loc[(df_new['label'] == "src to trgt - polyp")].index, num_src_to_trgt_to_cut, replace=False)
            df_minimized_backs = df_minimized_backs.drop(drop_indices_src_to_trgt_polyp)
            ###


            print(df_minimized_backs['label'].value_counts())

            return df_minimized_backs


    def _get_contr_feat_embeds(self, feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt=None):

        if feature_embedding_val_trgt is not None:

            concat_feats = torch.concat([feature_embedding_val_src, feature_embedding_val_src_in_trgt, feature_embedding_val_trgt], dim=0)
            concat_contrastive_embedding = self.contr_head(concat_feats)

            length_concat_contrastive_embedding = concat_contrastive_embedding.size(0)
            a_end_id = int(length_concat_contrastive_embedding/3)
            b_end_id = int(length_concat_contrastive_embedding/3 * 2)
            c_end_id = int(length_concat_contrastive_embedding/3 * 3)
            contr_feature_embedding_val_src = concat_contrastive_embedding[:a_end_id]
            contr_feature_embedding_val_src_in_trgt = concat_contrastive_embedding[a_end_id:b_end_id]
            contr_feature_embedding_val_trgt = concat_contrastive_embedding[b_end_id:c_end_id]

            return contr_feature_embedding_val_src, contr_feature_embedding_val_src_in_trgt, contr_feature_embedding_val_trgt
        else:

            concat_feats = torch.concat([feature_embedding_val_src, feature_embedding_val_src_in_trgt], dim=0)
            concat_contrastive_embedding = self.contr_head(concat_feats)

            length_concat_contrastive_embedding = concat_contrastive_embedding.size(0)
            a_end_id = int(length_concat_contrastive_embedding/2)
            b_end_id = int(length_concat_contrastive_embedding/2 * 2)
            contr_feature_embedding_val_src = concat_contrastive_embedding[:a_end_id]
            contr_feature_embedding_val_src_in_trgt = concat_contrastive_embedding[a_end_id:b_end_id]

            return contr_feature_embedding_val_src, contr_feature_embedding_val_src_in_trgt

    def _visualize_plots(self, mask_val, segmentation_mask_prediction_val, img_val):

        for idx in self.index_range:

            seg_mask = wandb.Image(F_vision.to_pil_image(mask_val[idx].float()).convert("L"))
            
            # Get the prediction output (normal and sigmoid) as wandb.Image for logging
            output_sigmoid = torch.sigmoid(segmentation_mask_prediction_val)
            output_segmap_sigmoid = wandb.Image(F_vision.to_pil_image(output_sigmoid[idx][0]).convert("L"))

            # Get the true images for logging
            input_image = wandb.Image(F_vision.to_pil_image(img_val[idx]))

            wandb.log({
                    # f"Feature Embedding {idx}": grid_array,
                    # f"Feature Embedding Sigmoid {idx}": grid_array_sigmoid,
                    # f"Coarse True Segmentation Mask {idx}": coarse_seg_mask,
                    f"True Segmentation Mask {idx}": seg_mask,
                    f"Prediction Output Sigmoid {idx}": output_segmap_sigmoid,
                    f"Input image {idx}": input_image,
                    })
            
            if img_val.size(0) <= idx:
                print()
                print("index would be out of range, so leaving")
                print()
                break