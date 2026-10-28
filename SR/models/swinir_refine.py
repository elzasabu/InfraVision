import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Window utilities
# ============================================================

def window_partition(x, window_size):
    """
    x: [B, H, W, C]
    returns: [num_windows * B, window_size, window_size, C]
    """

    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(
            -1,
            window_size,
            window_size,
            C
        )
    )

    return windows


def window_reverse(
    windows,
    window_size,
    H,
    W
):
    """
    windows:
        [num_windows * B, window_size, window_size, C]

    returns:
        [B, H, W, C]
    """

    B = int(
        windows.shape[0]
        / (H * W / window_size / window_size)
    )

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )

    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(
            B,
            H,
            W,
            -1
        )
    )

    return x


# ============================================================
# Window Attention
# ============================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size
    ):

        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.window_size = (
            window_size,
            window_size
        )

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        # ----------------------------------------------------
        # Relative position bias
        # ----------------------------------------------------

        relative_position_bias_table_size = (
            (2 * window_size - 1)
            *
            (2 * window_size - 1)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                relative_position_bias_table_size,
                num_heads
            )
        )

        # ----------------------------------------------------
        # Relative position index
        # ----------------------------------------------------

        coords_h = torch.arange(
            window_size
        )

        coords_w = torch.arange(
            window_size
        )

        coords = torch.stack(
            torch.meshgrid(
                coords_h,
                coords_w,
                indexing="ij"
            )
        )

        coords_flatten = torch.flatten(
            coords,
            1
        )

        relative_coords = (
            coords_flatten[:, :, None]
            -
            coords_flatten[:, None, :]
        )

        relative_coords = (
            relative_coords.permute(
                1,
                2,
                0
            ).contiguous()
        )

        relative_coords[:, :, 0] += (
            window_size - 1
        )

        relative_coords[:, :, 1] += (
            window_size - 1
        )

        relative_coords[:, :, 0] *= (
            2 * window_size - 1
        )

        relative_position_index = (
            relative_coords.sum(-1)
        )

        self.register_buffer(
            "relative_position_index",
            relative_position_index
        )

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=True
        )

        self.proj = nn.Linear(
            dim,
            dim
        )

        self.softmax = nn.Softmax(
            dim=-1
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02
        )


    def forward(
        self,
        x
    ):

        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                C // self.num_heads
            )
            .permute(
                2,
                0,
                3,
                1,
                4
            )
        )

        q, k, v = qkv[
            0
        ], qkv[
            1
        ], qkv[
            2
        ]

        q = q * self.scale

        attn = (
            q @ k.transpose(-2, -1)
        )

        # ----------------------------------------------------
        # Relative position bias
        # ----------------------------------------------------

        relative_position_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ]
        )

        relative_position_bias = (
            relative_position_bias.view(
                self.window_size[0]
                *
                self.window_size[1],

                self.window_size[0]
                *
                self.window_size[1],

                -1
            )
        )

        relative_position_bias = (
            relative_position_bias.permute(
                2,
                0,
                1
            ).contiguous()
        )

        attn = (
            attn
            +
            relative_position_bias.unsqueeze(0)
        )

        attn = self.softmax(
            attn
        )

        x = (
            attn @ v
        )

        x = (
            x.transpose(
                1,
                2
            )
            .reshape(
                B_,
                N,
                C
            )
        )

        x = self.proj(x)

        return x


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio=2
    ):

        super().__init__()

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.fc1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            dim
        )


    def forward(self, x):

        x = self.fc1(x)

        x = self.act(x)

        x = self.fc2(x)

        return x


# ============================================================
# Swin Transformer Block
# ============================================================

class SwinBlock(nn.Module):

    def __init__(
        self,
        dim=64,
        num_heads=4,
        window_size=8,
        mlp_ratio=2,
        shift_size=0
    ):

        super().__init__()

        self.dim = dim

        self.window_size = window_size

        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio
        )


    def forward(
        self,
        x,
        H,
        W
    ):

        B, L, C = x.shape

        shortcut = x

        x = self.norm1(x)

        x = x.view(
            B,
            H,
            W,
            C
        )

        # ----------------------------------------------------
        # Shifted window
        # ----------------------------------------------------

        if self.shift_size > 0:

            shifted_x = torch.roll(
                x,
                shifts=(
                    -self.shift_size,
                    -self.shift_size
                ),
                dims=(1, 2)
            )

        else:

            shifted_x = x

        # ----------------------------------------------------
        # Partition windows
        # ----------------------------------------------------

        x_windows = window_partition(
            shifted_x,
            self.window_size
        )

        x_windows = x_windows.view(
            -1,
            self.window_size
            *
            self.window_size,
            C
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        attn_windows = self.attn(
            x_windows
        )

        # ----------------------------------------------------
        # Reverse windows
        # ----------------------------------------------------

        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C
        )

        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W
        )

        # ----------------------------------------------------
        # Reverse cyclic shift
        # ----------------------------------------------------

        if self.shift_size > 0:

            x = torch.roll(
                shifted_x,
                shifts=(
                    self.shift_size,
                    self.shift_size
                ),
                dims=(1, 2)
            )

        else:

            x = shifted_x

        x = x.view(
            B,
            H * W,
            C
        )

        # ----------------------------------------------------
        # Residual attention
        # ----------------------------------------------------

        x = shortcut + x

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        x = (
            x
            +
            self.mlp(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# Residual Swin Group
# ============================================================

class ResidualSwinGroup(nn.Module):

    def __init__(
        self,
        dim=64,
        depth=4,
        num_heads=4,
        window_size=8
    ):

        super().__init__()

        blocks = []

        for i in range(depth):

            blocks.append(
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=2,
                    shift_size=(
                        0
                        if i % 2 == 0
                        else window_size // 2
                    )
                )
            )

        self.blocks = nn.ModuleList(
            blocks
        )

        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )


    def forward(
        self,
        x,
        H,
        W
    ):

        shortcut = x

        for block in self.blocks:

            x = block(
                x,
                H,
                W
            )

        # ----------------------------------------------------
        # Convert tokens back to image
        # ----------------------------------------------------

        B, _, C = x.shape

        image = x.view(
            B,
            H,
            W,
            C
        )

        image = image.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        image = self.conv(
            image
        )

        image = image.permute(
            0,
            2,
            3,
            1
        ).contiguous()

        image = image.view(
            B,
            H * W,
            C
        )

        return shortcut + image


# ============================================================
# SwinIR-style Refinement Network
# ============================================================

class SwinIRRefinement(nn.Module):

    def __init__(
        self,
        in_channels=3,
        embed_dim=48,
        num_groups=4,
        depth=4,
        num_heads=4,
        window_size=8
    ):

        super().__init__()

        self.embed_dim = embed_dim

        self.window_size = window_size

        # ----------------------------------------------------
        # Shallow feature extraction
        # ----------------------------------------------------

        self.conv_first = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Residual Swin groups
        # ----------------------------------------------------

        self.groups = nn.ModuleList()

        for _ in range(num_groups):

            self.groups.append(
                ResidualSwinGroup(
                    dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_size
                )
            )

        # ----------------------------------------------------
        # Body convolution
        # ----------------------------------------------------

        self.body_conv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        self.conv_last = nn.Conv2d(
            embed_dim,
            3,
            kernel_size=3,
            padding=1
        )


    def forward(
        self,
        x
    ):

        B, C, H, W = x.shape

        # ----------------------------------------------------
        # Window-size safety
        #
        # 256 is divisible by 8, so normally no padding
        # is required for your dataset.
        # ----------------------------------------------------

        if (
            H % self.window_size != 0
            or
            W % self.window_size != 0
        ):

            raise RuntimeError(
                "Image dimensions must be divisible "
                f"by window size {self.window_size}. "
                f"Received {H} x {W}."
            )

        # ----------------------------------------------------
        # Shallow features
        # ----------------------------------------------------

        shallow = self.conv_first(
            x
        )

        # Convert to tokens

        features = shallow.permute(
            0,
            2,
            3,
            1
        ).contiguous()

        features = features.view(
            B,
            H * W,
            self.embed_dim
        )

        # ----------------------------------------------------
        # Swin groups
        # ----------------------------------------------------

        for group in self.groups:

            features = group(
                features,
                H,
                W
            )

        # ----------------------------------------------------
        # Convert back to image
        # ----------------------------------------------------

        features = features.view(
            B,
            H,
            W,
            self.embed_dim
        )

        features = features.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        # ----------------------------------------------------
        # Global residual
        # ----------------------------------------------------

        features = self.body_conv(
            features
        )

        features = (
            features
            +
            shallow
        )

        # ----------------------------------------------------
        # Predict correction
        # ----------------------------------------------------

        correction = self.conv_last(
            features
        )

        # ----------------------------------------------------
        # Residual refinement
        #
        # Swin learns only what EDSR is missing.
        # ----------------------------------------------------

        output = (
            x
            +
            correction
        )

        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        return output