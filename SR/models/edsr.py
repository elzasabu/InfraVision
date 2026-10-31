import torch
import torch.nn as nn


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels=64, res_scale=0.1):

        super().__init__()

        self.res_scale = res_scale

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.block(x)

        return x + residual * self.res_scale


# ============================================================
# EDSR
# ============================================================

class EDSR(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=64,
        num_blocks=8,
        global_residual=True
    ):

        super().__init__()

        # True  -> output = input + predicted detail
        #          (ssim_residual checkpoints, used by the Swin scripts)
        # False -> output = predicted image directly
        #          (older checkpoints: edsr_epoch_*.pth, edsr_model*.pth)
        self.global_residual = global_residual

        # Clamp the output to [0, 1]. Training turns this off: a clamped pixel gets no gradient.
        self.clamp_output = True

        # ----------------------------------------------------
        # Head
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Residual body
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(
                    channels=channels,
                    res_scale=0.1
                )
            )

        self.body = nn.Sequential(*blocks)

        self.body_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Head
        # ----------------------------------------------------

        head = self.head(x)

        # ----------------------------------------------------
        # Residual body
        # ----------------------------------------------------

        body = self.body(head)

        body = self.body_conv(body)

        # ----------------------------------------------------
        # Global residual connection
        # ----------------------------------------------------

        features = head + body

        # ----------------------------------------------------
        # Predict residual/detail
        # ----------------------------------------------------

        residual = self.tail(features)

        # ----------------------------------------------------
        # Add learned details to input (global_residual=True)
        # or use the prediction directly (global_residual=False).
        #
        # IMPORTANT:
        # This must match how the checkpoint was trained.
        # The wrong mode washes the image out or darkens it.
        # ----------------------------------------------------

        if self.global_residual:
            output = x + residual
        else:
            output = residual

        # Keep RGB values between 0 and 1

        if self.clamp_output:
            output = torch.clamp(
                output,
                0.0,
                1.0
            )

        return output