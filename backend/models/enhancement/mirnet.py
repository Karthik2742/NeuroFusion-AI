"""
NeuroFusion AI - MIRNet Enhancement Model
Learning Enriched Features for Real Image Restoration and Enhancement.

Reference: Zamir et al., "Learning Enriched Features for Real Image Restoration and Enhancement"
           ECCV 2020. https://arxiv.org/abs/2003.06792
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from backend.models.enhancement.base import BaseEnhancer


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class SpatialAttention(nn.Module):
    """Spatial attention using average and max pooling."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(cat))
        return x * attn


class DAU(nn.Module):
    """
    Dual Attention Unit.
    
    Combines channel attention and spatial attention in parallel branches.
    """

    def __init__(self, num_feat: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size // 2)
        self.channel_attn = ChannelAttention(num_feat)
        self.spatial_attn = SpatialAttention()
        self.conv_out = nn.Conv2d(num_feat * 2, num_feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.relu(self.conv1(x))
        feat = self.conv2(feat)

        ca = self.channel_attn(feat)
        sa = self.spatial_attn(feat)

        out = self.conv_out(torch.cat([ca, sa], dim=1))
        return out + x


class SKFF(nn.Module):
    """
    Selective Kernel Feature Fusion.
    
    Dynamically selects and fuses features from multiple resolution streams
    using attention-based weighting.
    """

    def __init__(self, num_feat: int, height: int = 2, reduction: int = 8):
        super().__init__()
        self.height = height
        d = max(int(num_feat / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_feat, d, 1, padding=0, bias=False),
            nn.PReLU(),
        )

        self.fcs = nn.ModuleList([
            nn.Conv2d(d, num_feat, kernel_size=1, stride=1, bias=False)
            for _ in range(height)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats: list[torch.Tensor]) -> torch.Tensor:
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        return feats_V


class DownSample(nn.Module):
    """Downsample via strided convolution."""

    def __init__(self, in_channels: int, scale_factor: int, kernel_size: int = 3):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.PixelUnshuffle(scale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSample(nn.Module):
    """Upsample via pixel shuffle."""

    def __init__(self, in_channels: int, scale_factor: int, kernel_size: int = 3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // (scale_factor ** 2), kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class MRB(nn.Module):
    """
    Multi-scale Residual Block.
    
    Processes features at multiple resolutions simultaneously
    and fuses them using SKFF.
    """

    def __init__(self, num_feat: int, kernel_size: int = 3, height: int = 3, width: int = 2):
        super().__init__()
        self.height = height
        self.width = width

        # DAU blocks for each resolution stream
        self.dau_blocks = nn.ModuleList()
        for i in range(height):
            stream_blocks = nn.ModuleList()
            for j in range(width):
                stream_blocks.append(DAU(num_feat, kernel_size))
            self.dau_blocks.append(stream_blocks)

        # SKFF for feature fusion at each column
        self.skff_blocks = nn.ModuleList()
        for j in range(width):
            self.skff_blocks.append(SKFF(num_feat, height))

        # Down/Up sampling between resolutions
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(height - 1):
            self.down.append(
                nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=2, padding=1)
            )
            self.up.append(
                nn.Sequential(
                    nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create multi-scale copies
        copies = [x]
        for i in range(self.height - 1):
            copies.append(self.down[i](copies[-1]))

        # Process through DAU blocks
        for j in range(self.width):
            for i in range(self.height):
                copies[i] = self.dau_blocks[i][j](copies[i])

            # SKFF fusion
            # Upsample lower resolutions to match the highest resolution
            skff_input = [copies[0]]
            for i in range(1, self.height):
                upsampled = copies[i]
                for k in range(i):
                    upsampled = self.up[k](upsampled)
                # Match spatial dimensions
                if upsampled.shape[2:] != copies[0].shape[2:]:
                    upsampled = F.interpolate(
                        upsampled, size=copies[0].shape[2:], mode='bilinear', align_corners=False
                    )
                skff_input.append(upsampled)

            copies[0] = self.skff_blocks[j](skff_input)

        return copies[0] + x


class RRG(nn.Module):
    """Recursive Residual Group: multiple MRBs with residual connection."""

    def __init__(self, num_feat: int, num_mrb: int, kernel_size: int = 3,
                 height: int = 3, width: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MRB(num_feat, kernel_size, height, width) for _ in range(num_mrb)]
        )
        self.conv = nn.Conv2d(num_feat, num_feat, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.blocks(x)) + x


class MIRNet(BaseEnhancer):
    """
    MIRNet: Learning Enriched Features for Real Image Restoration.
    
    Multi-scale architecture with:
    - Dual Attention Units (DAU) for channel + spatial attention
    - Selective Kernel Feature Fusion (SKFF) for multi-resolution fusion
    - Recursive Residual Groups (RRG) for deep feature extraction
    
    Adapted for single-channel MRI enhancement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_feat: int = 64,
        kernel_size: int = 3,
        n_rrg: int = 3,
        n_mrb: int = 2,
        height: int = 3,
        width: int = 2,
    ):
        super().__init__(model_name="MIRNet")

        self.conv_in = nn.Conv2d(
            in_channels, n_feat, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.body = nn.Sequential(
            *[RRG(n_feat, n_mrb, kernel_size, height, width) for _ in range(n_rrg)]
        )

        self.conv_out = nn.Conv2d(
            n_feat, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

        logger.info(
            f"MIRNet initialized: n_feat={n_feat}, n_rrg={n_rrg}, "
            f"n_mrb={n_mrb}, height={height}, width={width}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual learning.
        
        Args:
            x: Input (B, C, H, W)
            
        Returns:
            Enhanced (B, C, H, W)
        """
        # Pad to even dimensions for pixel shuffle operations
        _, _, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        residual = x

        out = self.conv_in(x)
        out = self.body(out)
        out = self.conv_out(out) + residual

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out
