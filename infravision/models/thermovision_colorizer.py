"""ThermoVision U-Net colorizer: thermal (1 ch) -> RGB (3 ch), inference only.

Vendored from the ThermoVision AI project (`unet_models.py`, same author as
InfraVision, released here under this repository's MIT licence). Only the
ResNet34 U-Net used by the exp6/exp7 checkpoints is kept; the VGG16 encoder,
ImageNet weight loading and all training code were removed. The ResNet34
layers come from torchvision (BSD-3-Clause) and are built without weights,
so nothing is downloaded.

Model contract (what the checkpoints were trained with):
    input   N x 1 x H x W float in [0, 1] (8-bit thermal preview / 255),
            whole frames resized to 256 x 256; standardised inside forward()
            with mean 0.449 / std 0.226
    output  N x 3 x H x W RGB in [0, 1] (sigmoid)
    H, W    any multiple of 32 works (fully convolutional, bilinear skips),
            but the weights were only trained at 256 x 256

Checkpoint format written by ThermoVision:
    {"format_version": 1,
     "model_config": {"architecture": "pretrained_unet", "encoder": "resnet34", "freeze_encoder_bn": true},
     "model_state_dict": {...}, "run_args": {...}, "step": int, "val_total": float}
Generator weights only: no optimizer, EMA or discriminator state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226   # one grey channel: mean of the ImageNet channel means / stds
ARCHITECTURE = "pretrained_unet"
MULTIPLE = 32                                 # ResNet34 total stride


def conv_bn_relu(i, o):
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class ResNet34Encoder(nn.Module):
    channels = (64, 64, 128, 256, 512)  # strides 2, 4, 8, 16, 32

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet34(weights=None)  # architecture only; no download
        r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


class ThermoVisionUNet(nn.Module):
    """ThermoVision `PretrainedUNet` with the ResNet34 encoder (same state_dict keys)."""

    def __init__(self, encoder="resnet34", decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        if encoder != "resnet34":
            raise ValueError(f"only the resnet34 encoder is vendored, checkpoint asks for {encoder!r}")
        self.encoder = ResNet34Encoder()
        ec = self.encoder.channels
        d4, d3, d2, d1, d0 = decoder_channels
        self.dec4 = nn.Sequential(*conv_bn_relu(ec[4] + ec[3], d4), *conv_bn_relu(d4, d4))
        self.dec3 = nn.Sequential(*conv_bn_relu(d4 + ec[2], d3), *conv_bn_relu(d3, d3))
        self.dec2 = nn.Sequential(*conv_bn_relu(d3 + ec[1], d2), *conv_bn_relu(d2, d2))
        self.dec1 = nn.Sequential(*conv_bn_relu(d2 + ec[0], d1), *conv_bn_relu(d1, d1))
        self.stem = nn.Sequential(*conv_bn_relu(1, 16), *conv_bn_relu(16, 16))      # full-resolution thermal branch
        self.dec0 = nn.Sequential(*conv_bn_relu(d1 + 16, d0), *conv_bn_relu(d0, d0))
        self.head = nn.Conv2d(d0, 3, 3, padding=1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        z = (x - IMAGENET_MEAN) / IMAGENET_STD
        f0, f1, f2, f3, f4 = self.encoder(z)
        d = self.dec4(torch.cat([self._up(f4, f3), f3], 1))
        d = self.dec3(torch.cat([self._up(d, f2), f2], 1))
        d = self.dec2(torch.cat([self._up(d, f1), f1], 1))
        d = self.dec1(torch.cat([self._up(d, f0), f0], 1))
        d = self.dec0(torch.cat([self._up(d, x), self.stem(x)], 1))
        return torch.sigmoid(self.head(d))

    def config(self):
        return {"architecture": ARCHITECTURE, "encoder": "resnet34", "freeze_encoder_bn": True}


def is_thermovision_checkpoint(ck):
    return isinstance(ck, dict) and isinstance(ck.get("model_config"), dict) \
        and ck["model_config"].get("architecture") == ARCHITECTURE


def save_checkpoint(path, model, **extra):
    """Write a checkpoint in the ThermoVision format (used by the tests)."""
    torch.save({"format_version": 1, "model_config": model.config(), "model_state_dict": model.state_dict(),
                **extra}, path)
