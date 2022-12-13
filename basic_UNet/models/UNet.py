from pyexpat import model
import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from torchvision import models

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from pytorch_lightning import LightningModule

import torchmetrics

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class UNET_ResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        # self.base_model.load_state_dict(state_dict)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) #Sequential((0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)(1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(2): ReLU(inplace=True))
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class UNet(nn.Module):
    """Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation.
    Link - https://arxiv.org/abs/1505.04597
    >>> UNet(num_classes=2, num_layers=3)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    UNet(
      (layers): ModuleList(
        (0): DoubleConv(...)
        (1): Down(...)
        (2): Down(...)
        (3): Up(...)
        (4): Up(...)
        (5): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """

    def __init__(self, num_classes: int = 19, num_layers: int = 5, features_start: int = 64, bilinear: bool = False):
        """
        Args:
            num_classes: Number of output classes required (default 19 for KITTI dataset)
            num_layers: Number of layers in each side of U-net
            features_start: Number of features in first layer
            bilinear: Whether to use bilinear interpolation or transposed convolutions for upsampling.
        """
        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(3, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class UNet_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], bilinear=False):
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # U-Net Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # U-Net Up part
        for feature in reversed(features):
            # in the original paper they used bilinear upsampling, but ConvTransposed2d should work better

            if bilinear:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(feature*2, feature, kernel_size=1),
                    )
                )
            else:
                self.ups.append(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(DoubleConv(features*2, features))

        # creates the final feature map (latent space) in the U-Net
        self.bottlneck = DoubleConv(feature[-1], features[-1]*2)

        # output layer
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottlneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        return self.output_conv(x)


class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU (3x3 conv -> BN -> ReLU) ** 2.
    >>> DoubleConv(4, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DoubleConv(
      (conv): Sequential(...)
    )
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Combination of MaxPool2d and DoubleConv in series.
    >>> Down(4, 8)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Down(
      (net): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): DoubleConv(
          (net): Sequential(...)
        )
      )
    )
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by double 3x3 convolution.
    >>> Up(8, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Up(
      (upsample): ConvTranspose2d(8, 4, kernel_size=(2, 2), stride=(2, 2))
      (conv): DoubleConv(
        (net): Sequential(...)
      )
    )
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)




def createDeepLabv3():
    """DeepLabv3 class with custom head
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True,
                                                    )
    
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model.classifier = DeepLabHead(2048, num_classes=1) # num_classes is equal to output channels
    # Set the model in training mode
    # print(model)
    return model

class SegModel(LightningModule):
    """Semantic Segmentation Module.
    This is a basic semantic segmentation module implemented with Lightning.
    SegModel(
      (net): UNet(
        (layers): ModuleList(
          (0): DoubleConv(...)
          (1): Down(...)
          (2): Down(...)
          (3): Up(...)
          (4): Up(...)
          (5): Conv2d(64, 19, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int = 1,
        lr: float = 1e-3,
        num_layers: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.dice_coeff = torchmetrics.Dice()
        self.seg_metric_test = torchmetrics.Dice()
        self.seg_metric_test_during_val = torchmetrics.Dice()


        if model_type == "unet":
            self.net = UNet(
                num_classes=self.num_classes, num_layers=self.num_layers, features_start=self.features_start, bilinear=self.bilinear
            )
        
        if model_type == "unet_resnet_backbone":
            self.net = UNET_ResNet18(num_classes)

        elif model_type == "deeplabv3":
            self.net = createDeepLabv3()

        else:
            assert False, "No model selected"


        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):

        if self.model_type == "deeplabv3":
            return self.net(x)['out']

        else:
            return self.net(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        # return [opt], [sch]
        return opt

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()

        out = self(img)# needed when we use DeepLabv3 model   

        loss = F.binary_cross_entropy_with_logits(out, mask)
        log_dict = {"Binary Cross Entropy Loss": loss}

        # Log loss and metric
        self.log("Training Loss - Source (Binary Cross Entropy)", loss)

        return {"loss": loss, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        
        img, mask = batch["loader_val_source"]
        img = img.float()

        seg_out = self(img)# needed when we use DeepLabv3 model   

        loss_val = F.binary_cross_entropy_with_logits(seg_out, mask)
        self.dice_coeff.update(seg_out, mask.to(dtype=torch.uint8))

        # Log loss, metric and domain output
        self.log('Validation Loss (BCE)', loss_val, on_step=False, prog_bar=True, on_epoch=True)


        # Target/Test Data
        img_test, mask_test = batch["loader_target_test"]
        img_test = img_test.float()

        segmentation_mask_prediction_test = self(img_test)

        self.seg_metric_test_during_val.update(segmentation_mask_prediction_test, mask_test.to(dtype=torch.uint8))


        return {"Validation Loss (BCE)": loss_val}

    def validation_epoch_end(self, outs):

        dice = self.dice_coeff.compute()
        print()
        print("Dice: ", dice)
        print()
        # self.log("Validation Loss (Binary Cross Entropy)", loss, prog_bar=True)
        self.log("Dice Score (Validation)", dice, prog_bar=True)
        self.dice_coeff.reset()

        dice_test_during_val = self.seg_metric_test_during_val.compute()
        self.log("Dice Score on Test/Target (Validation)", dice_test_during_val, prog_bar=True)
        self.seg_metric_test_during_val.reset()


    def test_step(self, batch, batch_idx):
        
        img, mask = batch
        img = img.float()

        seg_out = self(img)   

        self.seg_metric_test.update(seg_out, mask.to(dtype=torch.uint8))


    def test_epoch_end(self, outs):

        seg_metric_test = self.seg_metric_test.compute()
        print("seg_metric_test", seg_metric_test)
        self.log("FINAL Dice Score on Target Test Data", seg_metric_test, prog_bar=True)
        self.seg_metric_test.reset()