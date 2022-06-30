import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


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


class UNET_(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], bilinear=False):
        super(UNET_, self).__init__()
        
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
