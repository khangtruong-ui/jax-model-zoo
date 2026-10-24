"""Building blocks for DiffusionDiff architecture in Flax Linen (jax.nn)."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def get_valid_group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    """Find the largest divisor of num_channels <= max_groups matching sid_unet convention."""
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return num_groups


def sinusoidal_embedding(
    values: jax.Array,
    dim: int,
    max_period: float = 10000.0,
) -> jax.Array:
    """
    Compute sinusoidal positional embeddings for 1D scalar values (timesteps or sigma).
    Exact formula matching PyTorch DiffusionDiff implementation.
    """
    if dim <= 0:
        return jnp.empty((values.shape[0], 0), dtype=values.dtype)

    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = values[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
    return embedding


def expand_to_spatial(embeddings: jax.Array, height: int, width: int) -> jax.Array:
    """
    Expand [B, dim] embedding vector spatially across (H, W) in Flax NHWC layout:
    [B, dim] -> [B, H, W, dim]
    """
    return jnp.broadcast_to(embeddings[:, None, None, :], (embeddings.shape[0], height, width, embeddings.shape[-1]))


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with normalization and SiLU activation."""

    in_channels: int
    out_channels: int
    norm_type: str = "batchnorm"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        res = x
        if self.in_channels != self.out_channels:
            res = nn.Conv(self.out_channels, kernel_size=(1, 1), use_bias=False, name="shortcut_conv")(x)
            if self.norm_type == "batchnorm":
                res = nn.BatchNorm(use_running_average=deterministic, name="shortcut_norm")(res)
            elif self.norm_type == "groupnorm":
                res = nn.GroupNorm(num_groups=min(32, self.out_channels), name="shortcut_norm")(res)

        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", use_bias=False, name="conv1")(x)
        if self.norm_type == "batchnorm":
            h = nn.BatchNorm(use_running_average=deterministic, name="norm1")(h)
        elif self.norm_type == "groupnorm":
            h = nn.GroupNorm(num_groups=min(32, self.out_channels), name="norm1")(h)
        h = jax.nn.silu(h)

        if not deterministic and self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)

        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", use_bias=False, name="conv2")(h)
        if self.norm_type == "batchnorm":
            h = nn.BatchNorm(use_running_average=deterministic, name="norm2")(h)
        elif self.norm_type == "groupnorm":
            h = nn.GroupNorm(num_groups=min(32, self.out_channels), name="norm2")(h)

        return jax.nn.silu(h + res)


class DecoderUpsampleStage(nn.Module):
    """Upsampling stage that increases spatial resolution by 2x."""

    in_channels: int
    out_channels: int
    upsample_mode: str = "bilinear"
    norm_type: str = "batchnorm"
    dropout_rate: float = 0.0
    num_res_blocks: int = 1

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        b, h, w, c = x.shape
        # 2x upsample
        if self.upsample_mode == "nearest":
            up = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
            up = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", use_bias=False, name="up_conv")(up)
        else:  # bilinear
            up = jax.image.resize(x, (b, h * 2, w * 2, c), method="bilinear")
            up = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", use_bias=False, name="up_conv")(up)

        if self.norm_type == "batchnorm":
            up = nn.BatchNorm(use_running_average=deterministic, name="norm")(up)
        elif self.norm_type == "groupnorm":
            up = nn.GroupNorm(num_groups=min(32, self.out_channels), name="norm")(up)
        x = jax.nn.silu(up)

        for i in range(self.num_res_blocks):
            x = ResidualConvBlock(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                norm_type=self.norm_type,
                dropout_rate=self.dropout_rate,
                name=f"res_blocks_{i}",
            )(x, deterministic=deterministic)

        return x


class SkipFusion(nn.Module):
    """UNet-style skip connection fusion block."""

    dec_channels: int
    skip_channels: int
    out_channels: Optional[int] = None
    norm_type: str = "batchnorm"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, dec_feat: jax.Array, skip_feat: jax.Array, deterministic: bool = True) -> jax.Array:
        out_ch = self.out_channels or self.dec_channels
        b, h, w, _ = dec_feat.shape

        if skip_feat.shape[1:3] != (h, w):
            skip_feat = jax.image.resize(skip_feat, (b, h, w, skip_feat.shape[-1]), method="bilinear")

        if self.norm_type == "batchnorm":
            skip_feat = nn.BatchNorm(use_running_average=deterministic, name="skip_norm")(skip_feat)
        elif self.norm_type == "groupnorm":
            skip_feat = nn.GroupNorm(num_groups=min(32, skip_feat.shape[-1]), name="skip_norm")(skip_feat)

        fused = jnp.concatenate([dec_feat, skip_feat], axis=-1)
        h_fused = nn.Conv(out_ch, kernel_size=(3, 3), padding="SAME", use_bias=False, name="conv")(fused)
        if self.norm_type == "batchnorm":
            h_fused = nn.BatchNorm(use_running_average=deterministic, name="norm")(h_fused)
        elif self.norm_type == "groupnorm":
            h_fused = nn.GroupNorm(num_groups=min(32, out_ch), name="norm")(h_fused)
        h_fused = jax.nn.silu(h_fused)

        if not deterministic and self.dropout_rate > 0.0:
            h_fused = nn.Dropout(rate=self.dropout_rate)(h_fused, deterministic=deterministic)

        if h_fused.shape == dec_feat.shape:
            return dec_feat + h_fused
        return h_fused


class PerpendicularSkipFusion(nn.Module):
    """
    Perpendicular Skip Connection Fusion Block for DiffusionDiff-V2 and Minimized.
    Fuses features from parallel frozen VAE decoder into the trainable latent decoder.
    """

    dec_channels: int
    perp_channels: int
    out_channels: Optional[int] = None
    norm_type: str = "batchnorm"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, dec_feat: jax.Array, perp_feat: jax.Array, deterministic: bool = True) -> jax.Array:
        out_ch = self.out_channels or self.dec_channels
        b, h, w, _ = dec_feat.shape

        if perp_feat.shape[1:3] != (h, w):
            perp_feat = jax.image.resize(perp_feat, (b, h, w, perp_feat.shape[-1]), method="bilinear")

        if self.norm_type == "batchnorm":
            perp_feat = nn.BatchNorm(use_running_average=deterministic, name="perp_norm")(perp_feat)
        elif self.norm_type == "groupnorm":
            perp_feat = nn.GroupNorm(num_groups=min(32, perp_feat.shape[-1]), name="perp_norm")(perp_feat)

        fused = jnp.concatenate([dec_feat, perp_feat], axis=-1)
        h_fused = nn.Conv(out_ch, kernel_size=(3, 3), padding="SAME", use_bias=False, name="conv")(fused)
        if self.norm_type == "batchnorm":
            h_fused = nn.BatchNorm(use_running_average=deterministic, name="norm")(h_fused)
        elif self.norm_type == "groupnorm":
            h_fused = nn.GroupNorm(num_groups=min(32, out_ch), name="norm")(h_fused)
        h_fused = jax.nn.silu(h_fused)

        if not deterministic and self.dropout_rate > 0.0:
            h_fused = nn.Dropout(rate=self.dropout_rate)(h_fused, deterministic=deterministic)

        if h_fused.shape == dec_feat.shape:
            return dec_feat + h_fused
        return h_fused


class AuxiliaryClassifier(nn.Module):
    """Auxiliary binary/multiclass classification head on high-dimensional Z representation."""

    in_channels: int
    num_classes: int = 2
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z: jax.Array, deterministic: bool = True) -> jax.Array:
        # z is [B, H_z, W_z, C]
        pooled = jnp.mean(z, axis=(1, 2))  # [B, C]
        h = nn.Dense(128, name="fc1")(pooled)
        h = jax.nn.silu(h)
        if not deterministic and self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(self.num_classes, name="fc2")(h)
