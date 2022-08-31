import torch
import numpy as np
from torch import _nnpack_available
import torch.nn.functional as F
import pytorch_lightning as pl

from models.UNet import UNET_ResNet18

from utils.FFT import FDA_source_to_target
from utils.segmentation_metrics import Dice_Coefficient

class FDA_first_train(pl.LightningModule):

    def __init__(
        self,
        num_classes: int=1,
        lr: float = 1e-3,
        LB: float = 0.1,
        eta: float = 2.0,
        entW: float = 0.005,
):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.net = UNET_ResNet18(self.num_classes)
        self.LB = LB
        self.eta = eta
        self.entW = entW
        self.dice_coeff = Dice_Coefficient()

        self.IMG_MEAN = torch.reshape(torch.from_numpy(np.array((0.3052, 0.2198, 0.1685), dtype=np.float32)), (1,3,1,1))

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):

        return self.net(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        # return [opt], [sch]
        return opt

    def training_step(self, batch, batch_nb):

        # 0. Load images as batches
        batch_source = batch["loader_source"]
        batch_target = batch["loader_target"]

        img_source, mask_source = batch_source
        img_target, _ = batch_target

        #----------------------------------------------------------------------------------------#
        # 1. transforming the source image to the target style as in FDA by Yang et al. 
        src_in_trg = FDA_source_to_target(img_source, img_target, L=self.LB)

        # 2. substract mean

        B, _, H, W = img_source.shape
        mean_img = self.IMG_MEAN.repeat(B,1,H,W).cuda()

        src_img = src_in_trg.clone() - mean_img
        trg_img = img_target.clone() - mean_img

        #----------------------------------------------------------------------------------------#

        # 3. send src and trg image batches to the model
        out = self(src_img)

        # 4. Compute the Binary cross entropy loss for source image with target style
        loss_seg = F.binary_cross_entropy_with_logits(out, mask_source)

        # 5. Compute the Entropy Loss from the target image
        ## Is it needed to calculate softmax etc. if it is 1 class only anyway? --> I apply sigmoid here/Have a look on how the out values look like!!
        out = self(trg_img) # shape of out is: B, 1 (N_Classes), 256 (H), 256 (W)
        # P = F.softmax(out, dim=1)        # [B, 19, H, W] (from paper), in our case: [B, 1, H, W] 
        P = F.sigmoid(out)
        # logP = F.log_softmax(out, dim=1) # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        logP = F.logsigmoid(out)
        PlogP = P * logP               # [B, 19, H, W] (from paper), in our case: [B, 1, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W] (from paper)
        # ent = ent / 2.9444         # chanage when classes is not 19 (from paper), we leave out this line?? 
        
        # compute robust entropy/Charbonnier penalty function
        ent = ent ** 2.0 + 1e-8
        ent = ent ** self.eta
        loss_ent = ent.mean()

        # 6 Compute the overall loss
        loss = loss_seg + self.entW * loss_ent

        self.log('Binary Cross Entropy Loss Source Images', loss_seg, prog_bar=True)
        self.log('Entropy Loss Target Images', loss_ent, prog_bar=True)
        self.log('Overall Loss', loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):

        batch_source = batch["loader_val_source"]
        batch_target = batch["loader_target_test"]

        img, mask = batch_source

        seg_out = self(img)

        loss_val = F.binary_cross_entropy_with_logits(seg_out, mask)
        dice_coeff_values = self.dice_coeff(seg_out, mask)
        curr_mean_dice = torch.mean(dice_coeff_values[:, -2], dim=0)

        self.log('Validation Loss - Source Data', loss_val, prog_bar=True, on_epoch=True)
        self.log('Dice Score - Source Data', curr_mean_dice, prog_bar=True,on_epoch=True)
       
        img_test, mask_test = batch_target

        seg_out_test = self(img_test)

        dice_coeff_values_test = self.dice_coeff(seg_out_test, mask_test)
        curr_mean_dice_test = torch.mean(dice_coeff_values_test[:, -2], dim=0) 

        self.log("Dice Score on Target Test Data", curr_mean_dice_test, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"val_loss": loss_val}

    def test_step(self, batch, batch_idx):

        img, mask = batch

        seg_out, _ = self(img)

        dice_coeff_values = self.dice_coeff(seg_out, mask)
        curr_mean_dice = torch.mean(dice_coeff_values[:, -2], dim=0)

        self.log("FINAL Dice Score on Target Test Data", curr_mean_dice, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"curr_mean_dice": curr_mean_dice}


class UNet_baseline(FDA_first_train):


    def training_step(self, batch, batch_nb):
        
        # 0. Load images as batches
        batch_source = batch["loader_source"]
        batch_target = batch["loader_target"]

        img_source, mask_source = batch_source
        img_target, _ = batch_target

        out = self(img_source)
        loss = F.binary_cross_entropy_with_logits(out, mask_source)

        self.log('Overall Loss', loss, prog_bar=True)

        return loss