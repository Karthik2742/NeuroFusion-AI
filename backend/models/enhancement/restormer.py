"""
NeuroFusion AI - Restormer Enhancement Model
Efficient Transformer for High-Resolution MRI Enhancement.

Reference: Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration"
           CVPR 2022. https://arxiv.org/abs/2111.09881
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from backend.models.enhancement.base import BaseEnhancer


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (MDTA).
    
    Core attention mechanism of Restormer that applies self-attention
    across channels instead of spatial dimensions, reducing complexity
    from O(N²) to O(N·C).
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            padding=1, groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN).
    
    Uses depth-wise convolutions and gating for efficient
    feature transformation in the feed-forward layer.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D feature maps (BiasFree or WithBias)."""

    def __init__(self, dim: int, layer_norm_type: str = "WithBias"):
        super().__init__()
        self.body = nn.LayerNorm(dim)
        self.layer_norm_type = layer_norm_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return self.body(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TransformerBlock(nn.Module):
    """Single Restormer Transformer Block: LayerNorm → MDTA → LayerNorm → GDFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DownSample(nn.Module):
    """Downsample with pixel unshuffle."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpSample(nn.Module):
    """Upsample with pixel shuffle."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(BaseEnhancer):
    """
    Restormer: Efficient Transformer for High-Resolution Image Restoration.
    
    Adapted for single-channel MRI enhancement.
    Uses Multi-Dconv Head Transposed Attention (MDTA) and 
    Gated-Dconv Feed-Forward Network (GDFN) in a U-shaped architecture.
    
    Architecture:
        Encoder (4 levels) → Latent → Decoder (4 levels) with skip connections
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_heads: list[int] | None = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ):
        super().__init__(model_name="Restormer")

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        # Input projection
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

        # Encoder
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, num_heads[0], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[0])]
        )
        self.down1_2 = DownSample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[1], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[1])]
        )
        self.down2_3 = DownSample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, num_heads[2], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[2])]
        )
        self.down3_4 = DownSample(dim * 4)

        # Latent
        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, num_heads[3], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[3])]
        )

        # Decoder
        self.up4_3 = UpSample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, num_heads[2], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[2])]
        )

        self.up3_2 = UpSample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[1], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[1])]
        )

        self.up2_1 = UpSample(dim * 2)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[0], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[0])]
        )

        # Refinement
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim * 2, num_heads[0], ffn_expansion_factor, bias, layer_norm_type)
              for _ in range(num_blocks[0])]
        )

        # Output projection
        self.output = nn.Conv2d(
            dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        logger.info(f"Restormer initialized: dim={dim}, blocks={num_blocks}, heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Enhanced tensor (B, C, H, W)
        """
        # Pad input to be divisible by 8
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        inp = x

        # Encoder
        inp_enc_level1 = self.patch_embed(inp)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Refinement
        out = self.refinement(out_dec_level1)
        out = self.output(out) + x  # Residual connection

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out
