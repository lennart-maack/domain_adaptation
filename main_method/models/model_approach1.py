"""

This approach uses:

1. Contrastive Learning, using only SegContrSegLos, with x^s, x^s->t in one minibatch
2. FDA as Style Transfer Head
3. No generation of pseudo labels (yet)

"""

import torch
from torch import nn
from torch.autograd import forward_ad
import torch.nn.functional as F
import PIL
import cv2
import seaborn as sns

import pytorch_lightning as pl

from main_method.utils.visualisation import create_tsne
from main_method.utils.losses import PixelContrastLoss

from main_method.models.resnet.resnet_backbone import ResNetBackbone

from torchmetrics import Dice

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
from other_domain_adapt_methods.utils.FFT import FDA_source_to_target


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

class MainNetwork(pl.LightningModule):
    def __init__(
                self,
                wandb_configs
    ):
        super().__init__()

        self.contr_head_type = wandb_configs.contr_head_type

        self.learning_rate = wandb_configs.lr
        self.seg_metric = Dice(num_classes=2, average="macro")
        self.seg_metric_test = Dice(num_classes=2, average="macro")
        self.seg_metric_test_during_val = Dice(num_classes=2, average="macro")
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cross_entropy_loss_pseudo = torch.nn.CrossEntropyLoss(reduce=False)
        self.sup_contr_loss = PixelContrastLoss(temperature=wandb_configs.temperature, base_temperature=wandb_configs.base_temperature, max_samples=wandb_configs.max_samples, max_views=wandb_configs.max_views)
        self.visualize_tsne = wandb_configs.visualize_tsne

        ##########################################################
        # Set the resNet Backbone + full Decoder
        self.resnet_backbones = ResNetBackbone()
        if wandb_configs.model_type == "dilated":
            print("Using a dilated ResNet18 backbone", "\n")
            self.encoder = self.resnet_backbones.deepbase_resnet18_dilated8()
            self.decoder = Dilated_Decoder()

        elif wandb_configs.model_type == "normal":
            print("Using a normal ResNet18 backbone", "\n")
            self.encoder = self.resnet_backbones.resnet18()
            self.decoder = Normal_Decoder()
            
        else:
            raise NotImplementedError("Choose a valid model_type")


        ###########################################################
        # Choose if FDA should be used
        self.apply_FDA = wandb_configs.apply_FDA

        ###########################################################
        # Choose if self learning should be used
        self.use_self_learning = wandb_configs.use_self_learning 
        self.use_target_for_contr = wandb_configs.use_target_for_contr

        ###########################################################
        # Choosing the contrastive head type to use
        if wandb_configs.contr_head_type == "no_contr_head":
            print("Using No contrastive head/loss")
            self.contr_head = None
        else:
            print("Using contrastive head: ", wandb_configs.contr_head_type)
            self.contr_head = Contrastive_Head(512, head_type=wandb_configs.contr_head_type)
        
        self.contrastive_lambda = wandb_configs.contrastive_lambda

        ###########################################################
        # Index range which is used for the amount of images logged
        self.index_range = list(range(wandb_configs.index_range[0],wandb_configs.index_range[1]+1))

    def forward(self, input_data):

        output = self.encoder(input_data)

        layer0_output, maxpool_output, layer1_output, layer2_output, layer3_output, layer4_output = output

        segmentation_mask_prediction = self.decoder(input_data, layer0_output, layer1_output, layer2_output, layer3_output, layer4_output)
        
        return segmentation_mask_prediction, layer4_output
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.trainer.max_epochs, power=0.9)
        
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
        img_target, _ = batch_target
        

        # 1. Calculate CE loss - source img 
        segmentation_mask_prediction_src, layer4_output_src = self(img_source)
        loss_ce_src = self.cross_entropy_loss(segmentation_mask_prediction_src, mask_source)
        self.log("Training Loss - Source (Cross Entropy)", loss_ce_src, prog_bar=True)


        # 2. Calculate FDA transformed source img and FDA entropy loss
        if self.apply_FDA:

            src_in_trg = FDA_source_to_target(img_source, img_target, L=0.01)
            
            loss_ent = self._calculate_FDA_loss_ent(img_target)
            self.log("Training Entropy Loss (Charbonnier)", loss_ent, prog_bar=True)

            segmentation_mask_prediction_src_in_trgt, layer4_output_src_in_trgt = self(src_in_trg)
            loss_ce_src_in_trgt = self.cross_entropy_loss(segmentation_mask_prediction_src_in_trgt, mask_source)
            self.log("Training Loss - Source in Target (Cross Entropy)", loss_ce_src_in_trgt, prog_bar=True)

        else:
            loss_ce_src_in_trgt = 0
            loss_ent = 0

        # 3. Calculate CE loss - target img with pseudo_labels
        if self.use_self_learning:
            if self.current_epoch > 30:
                segmentation_mask_prediction_trgt, layer4_output_trgt = self(img_target)
                pseudo_mask_target = torch.argmax(segmentation_mask_prediction_trgt, 1)
                max_pred, max_indices = torch.max(segmentation_mask_prediction_trgt, 1)
                m_t = (max_pred > 0.9).long()
                initial_pseudo_target_loss = self.cross_entropy_loss_pseudo(segmentation_mask_prediction_trgt, pseudo_mask_target)
                confidence_pseudo_target_loss_matrix = torch.multiply(initial_pseudo_target_loss, m_t)
                numerator = torch.sum(confidence_pseudo_target_loss_matrix)
                denom = torch.count_nonzero(confidence_pseudo_target_loss_matrix)
                confidence_pseudo_target_loss = numerator/denom
                self.log("Training Loss - Target (Cross Entropy)", confidence_pseudo_target_loss, prog_bar=True)
            else:
                confidence_pseudo_target_loss = 0
        else:
            confidence_pseudo_target_loss = 0

        # 4. Calculate Contrastive Loss
        if self.contr_head_type != "no_contr_head":
            
            if self.apply_FDA:
                # Feature embeddings of source, sourceToTarget and target are to be used !!!!----> NEEDS TO BE FINISHED!!!!
                if self.use_self_learning and self.use_target_for_contr:

                    contrastive_loss = self._get_contr_loss(layer4_output_src, segmentation_mask_prediction_src, mask_source, 
                                                        layer4_output_src_in_trgt, segmentation_mask_prediction_src_in_trgt,
                                                        layer4_output_trgt, segmentation_mask_prediction_trgt, pseudo_mask_target, m_t)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)
                
                # Feature embeddings of source and sourceToTarget are to be used
                else:
                    contrastive_loss = self._get_contr_loss(layer4_output_src, segmentation_mask_prediction_src, mask_source, 
                                                            layer4_output_src_in_trgt, segmentation_mask_prediction_src_in_trgt)
                    self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)
            
            # Feature embeddings of source are to be used
            else:
                contrastive_loss = self._get_contr_loss(layer4_output_src, segmentation_mask_prediction_src, mask_source)
                self.log("Training Contrastive Loss", contrastive_loss, prog_bar=True)
        else:
            contrastive_loss = 0    
        

        overall_loss = loss_ce_src + loss_ce_src_in_trgt + 0.01 * loss_ent + confidence_pseudo_target_loss + self.contrastive_lambda * contrastive_loss
        self.log("Overall Loss Training", overall_loss, prog_bar=True)

        return overall_loss



    def validation_step(self, batch, batch_idx):

        # 1. Calculate CE Loss and Dice Score on source validation data
        batch_val = batch["loader_val_source"]
        img_val, mask_val = batch_val # mask.size(): [bs, 256, 256]
        
        segmentation_mask_prediction_val, feature_embedding_val = self(img_val)
        val_loss = self.cross_entropy_loss(segmentation_mask_prediction_val, mask_val)
        self.seg_metric.update(segmentation_mask_prediction_val, mask_val.to(dtype=torch.uint8))
        self.log("Validation CE Loss", val_loss, on_step=False, on_epoch=True)

        # 2. Calculate Dice Score on target test data
        batch_test = batch["loader_target_test"]
        img_test, mask_test = batch_test

        segmentation_mask_prediction_test, _ = self(img_test)
        self.seg_metric_test_during_val.update(segmentation_mask_prediction_test, mask_test.to(dtype=torch.uint8))

        # 3. Section for visualization
        if batch_idx == 0 and self.current_epoch % 4 == 0:
            
            if self.visualize_tsne:
                self._visualize_tsne(feature_embedding_val, mask_val, pca_n_comp=50, perplexity=30)
            
            self._visualize_plots(mask_val, segmentation_mask_prediction_val, img_val)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        
        dice = self.seg_metric.compute()
        self.log("Dice Score (Validation)", dice, prog_bar=True)
        self.seg_metric.reset()

        dice_test_during_val = self.seg_metric_test_during_val.compute()
        self.log("Dice Score on Test/Target (Validation)", dice_test_during_val, prog_bar=True)
        self.seg_metric_test_during_val.reset()


    def test_step(self, batch, batch_idx):

        img, mask = batch # mask.size(): [bs, 1, 256, 256]

        segmentation_mask_prediction, feature_embedding = self(img)
        self.seg_metric_test.update(segmentation_mask_prediction, mask.to(dtype=torch.uint8))        


    def test_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`

        seg_metric_test = self.seg_metric_test.compute()
        self.log("FINAL Dice Score on Target Test Data", seg_metric_test, prog_bar=True)
        self.seg_metric_test.reset()




    def _calculate_FDA_loss_ent(self, image_target):

        # 2. Compute the Entropy Loss from the target image (Needed for Regularization with Charbonnier penalty function)
        ## Is it needed to calculate softmax etc. if it is 1 class only anyway? --> I apply sigmoid here/Have a look on how the out values look like!!
        out = self(image_target)[0] # shape of out is: B, 1 (N_Classes), 256 (H), 256 (W)
        # P = F.softmax(out, dim=1)        # [B, 19, H, W] (from paper), in our case: [B, 1, H, W] 
        P = F.sigmoid(out)
        # logP = F.log_softmax(out, dim=1) # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        logP = F.logsigmoid(out)
        PlogP = P * logP               # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W] (from paper)
        # ent = ent / 2.9444         # chanage when classes is not 19 (from paper), we leave out this line?? 
        
        # compute robust entropy/Charbonnier penalty function
        ent = ent ** 2.0 + 1e-8
        eta = 2.0
        ent = ent ** eta
        loss_ent = ent.mean()

        return loss_ent


    def _get_contr_loss(self, layer4_output_src, segmentation_mask_prediction_src, mask_source, 
                        layer4_output_src_in_trgt=None, segmentation_mask_prediction_src_in_trgt=None,
                        layer4_output_trgt = None, segmentation_mask_prediction_trgt = None, pseudo_mask_target = None, m_t = None):

        if layer4_output_src_in_trgt is not None and segmentation_mask_prediction_src_in_trgt is not None:
            # 5. Use both the latent feature representation from source and source_in_target to create a concatenated latent feature representation
            ## Concat the two latent feature embeddings
            feature_embed_list = [layer4_output_src, layer4_output_src_in_trgt]
            concat_feature_embed = torch.cat(feature_embed_list, dim=0)
            ## Concat the two corresponding masks
            mask_list = [mask_source, mask_source]
            concat_masks = torch.cat(mask_list, dim=0)
            ## Concat the two predcitions
            pred_list = [segmentation_mask_prediction_src, segmentation_mask_prediction_src_in_trgt]
            concat_predictions = torch.cat(pred_list, dim=0)

            # downscale the prediction or use the coarse prediction could be implemented - until now we use the coarse prediction and the coarse mask
            contrastive_embedding = self.contr_head(concat_feature_embed)

            _, concat_predictions = torch.max(concat_predictions, 1)
            contrastive_loss = self.sup_contr_loss(contrastive_embedding, concat_masks, concat_predictions)

        else:
            contrastive_embedding_src = self.contr_head(layer4_output_src)
            predictions = torch.max(segmentation_mask_prediction_src, 1)
            contrastive_loss = self.sup_contr_loss(contrastive_embedding_src, mask_source, predictions)

        return contrastive_loss


    def _visualize_tsne(self, feature_embedding_val, mask_val, pca_n_comp=50, perplexity=30):

        df_tsne = create_tsne(feature_embedding_val, mask_val, pca_n_comp=pca_n_comp, perplexity=perplexity)
        fig = plt.figure(figsize=(10,10))
        sns_plot = sns.scatterplot(x="tsne-one", y="tsne-two", hue="label", data=df_tsne)
        fig = sns_plot.get_figure()
        fig.savefig(f"tSNE_vis_perplex_{perplexity}.jpg")
        tsne_img = cv2.imread(f"tSNE_vis_perplex_{perplexity}.jpg")
        tSNE_image = wandb.Image(PIL.Image.fromarray(tsne_img))

        wandb.log({f"tSNE visualization perplex {perplexity}": tSNE_image})



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
