"""Small educational U-Net. PyTorch is only necessary for the DL track."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.down1 = ConvBlock(1, width)
        self.down2 = ConvBlock(width, width * 2)
        self.mid = ConvBlock(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)
        # Residual learning: initialization reproduces the zero-filled input, so
        # short runs do not start from a near-constant gray output.
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool(skip1))
        hidden = self.mid(self.pool(skip2))
        hidden = self.dec2(torch.cat([self.up2(hidden), skip2], dim=1))
        hidden = self.dec1(torch.cat([self.up1(hidden), skip1], dim=1))
        return torch.clamp(x + self.out(hidden), 0.0, 1.0)
