import torch
import torch.nn as nn
import torch.nn.functional as F

# Número de canales de entrada MRI
# BraTS usa:
# T1, T1CE, T2, FLAIR
IN_CHANNELS = 4

# Número de clases de salida
# BraTS:
# 0 = background
# 1 = necrosis
# 2 = edema
# 3 = enhancing tumor
OUT_CLASSES = 4

# Número base de filtros
BASE_FILTERS = 32


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool3d(kernel_size=2),

            DoubleConv(
                in_channels,
                out_channels
            )
        )

    def forward(self, x):

        return self.block(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            [
                diffW // 2,
                diffW - diffW // 2,

                diffH // 2,
                diffH - diffH // 2,

                diffD // 2,
                diffD - diffD // 2
            ]
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        filters = BASE_FILTERS

        self.inc = DoubleConv(
            IN_CHANNELS,
            filters
        )

        self.down1 = Down(
            filters,
            filters * 2
        )

        self.down2 = Down(
            filters * 2,
            filters * 4
        )

        self.down3 = Down(
            filters * 4,
            filters * 8
        )

        self.bottleneck = Down(
            filters * 8,
            filters * 16
        )

        self.up1 = Up(
            filters * 16,
            filters * 8
        )

        self.up2 = Up(
            filters * 8,
            filters * 4
        )

        self.up3 = Up(
            filters * 4,
            filters * 2
        )

        self.up4 = Up(
            filters * 2,
            filters
        )

        self.outc = nn.Conv3d(
            filters,
            OUT_CLASSES,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        logits = self.outc(x)

        return logits


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet3D().to(device)

    print(model)

    x = torch.randn(
        1,
        IN_CHANNELS,
        128,
        128,
        128
    ).to(device)

    with torch.no_grad():

        y = model(x)

    print("Input shape :", x.shape)

    print("Output shape:", y.shape)