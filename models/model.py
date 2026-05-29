"""Configurable U-Net for semantic segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv(+BN)+ReLU blocks. Padding keeps spatial size unchanged."""

    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False) -> None:
        super().__init__()
        layers = []
        for i in range(2):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Decoder upsampling block with optional skip concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsampling: str = "bilinear",
        use_skip: bool = True,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        if upsampling == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        elif upsampling == "transpose":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise ValueError("upsampling must be 'bilinear' or 'transpose'")

        conv_in_channels = out_channels * 2 if use_skip else out_channels
        self.conv = DoubleConv(conv_in_channels, out_channels, batch_norm=batch_norm)

    @staticmethod
    def _match_spatial_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Match spatial size if odd dimensions ever create a mismatch.

        With 256x256 input and padded convolutions, cropping is not needed. This fallback
        keeps the implementation robust for other sizes.
        """
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip tensor is required when use_skip=True")
            x = self._match_spatial_size(x, skip)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with configurable upsampling, skip connections, and BatchNorm."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        base_channels: int = 64,
        upsampling: str = "bilinear",
        use_skip: bool = True,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        self.model_config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "base_channels": base_channels,
            "upsampling": upsampling,
            "use_skip": use_skip,
            "batch_norm": batch_norm,
        }

        self.enc1 = DoubleConv(in_channels, c, batch_norm=batch_norm)
        self.enc2 = DoubleConv(c, c * 2, batch_norm=batch_norm)
        self.enc3 = DoubleConv(c * 2, c * 4, batch_norm=batch_norm)
        self.enc4 = DoubleConv(c * 4, c * 8, batch_norm=batch_norm)
        self.bottom = DoubleConv(c * 8, c * 16, batch_norm=batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = UpBlock(c * 16, c * 8, upsampling=upsampling, use_skip=use_skip, batch_norm=batch_norm)
        self.up3 = UpBlock(c * 8, c * 4, upsampling=upsampling, use_skip=use_skip, batch_norm=batch_norm)
        self.up2 = UpBlock(c * 4, c * 2, upsampling=upsampling, use_skip=use_skip, batch_norm=batch_norm)
        self.up1 = UpBlock(c * 2, c, upsampling=upsampling, use_skip=use_skip, batch_norm=batch_norm)
        self.classifier = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)             # 256 x 256 x 64 for default settings
        e2 = self.enc2(self.pool(e1)) # 128 x 128 x 128
        e3 = self.enc3(self.pool(e2)) # 64 x 64 x 256
        e4 = self.enc4(self.pool(e3)) # 32 x 32 x 512
        b = self.bottom(self.pool(e4))# 16 x 16 x 1024

        d4 = self.up4(b, e4 if self.up4.use_skip else None)
        d3 = self.up3(d4, e3 if self.up3.use_skip else None)
        d2 = self.up2(d3, e2 if self.up2.use_skip else None)
        d1 = self.up1(d2, e1 if self.up1.use_skip else None)
        return self.classifier(d1)    # N x num_classes x 256 x 256


def build_model(model_cfg: dict, num_classes: int) -> UNet:
    return UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=num_classes,
        base_channels=int(model_cfg.get("base_channels", 64)),
        upsampling=str(model_cfg.get("upsampling", "bilinear")),
        use_skip=bool(model_cfg.get("use_skip", True)),
        batch_norm=bool(model_cfg.get("batch_norm", False)),
    )
