import torch
from torch import nn

import utils
from unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x: torch.Tensor, postprocess: bool = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return self.postprocess(logits) if postprocess else logits

    @classmethod
    def postprocess(cls, logits: torch.Tensor, threshold: float = 0.5):
        mask = (torch.softmax(logits, dim=1)[:, 1] > threshold)
        mask = utils.apply_connected_components_filter(mask)
        return mask

    @classmethod
    def to_image(cls, *args, **kwargs):
        return utils.to_image(*args, **kwargs)

