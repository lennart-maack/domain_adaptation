import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from metrics import Dice_Coefficient

from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead



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


class DoubleConv(nn.Module):
    """Double Convolution and BN and ReLU (3x3 conv -> BN -> ReLU) ** 2.
    >>> DoubleConv(4, 4)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DoubleConv(
      (net): Sequential(...)
    )
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


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


class SegModel(LightningModule):
    """Semantic Segmentation Module.
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
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
        model_name: str,
        num_classes: int = 1,
        lr: float = 1e-3,
        num_layers: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.lr = lr
        self.model_name = model_name
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.dice_coeff = Dice_Coefficient()

        if model_name == "unet":
            self.net = UNet(
                num_classes=self.num_classes, num_layers=self.num_layers, features_start=self.features_start, bilinear=self.bilinear
            )
        
        elif model_name == "deeplabv3":
            self.net = createDeepLabv3()

        else:
            assert False, "No model selected"


        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        
        if self.model_name == "unet":
            return self.net(x)
        
        elif self.model_name == "deeplabv3":
            return self.net(x)['out']

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
        log_dict = {"train_loss": loss}

        # Log loss and metric
        self.log('train_loss', loss)

        return {"loss": loss, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()

        out = self(img) # needed when we use DeepLabv3 model

        loss_val = F.binary_cross_entropy_with_logits(out, mask)
        dice_coeff_values = self.dice_coeff(out, mask)
        curr_mean_dice = torch.mean(dice_coeff_values[:, -2], dim=0)
        
        # Log loss and metric
        self.log('val_loss', loss_val, prog_bar=True)
        self.log('curr_mean_dice', curr_mean_dice, prog_bar=True)

        return {"val_loss": loss_val}