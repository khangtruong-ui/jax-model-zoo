"""EfficientNet-B2 backbone implemented in Flax Linen (jax.nn)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import EfficientNetConfig


def _round_channels(channels: int, width_mult: float, divisor: int = 8) -> int:
    """Round channel count to nearest multiple of divisor."""
    if not width_mult:
        return channels
    new_channels = channels * width_mult
    new_p = max(divisor, int(new_channels + divisor / 2) // divisor * divisor)
    if new_p < 0.9 * new_channels:
        new_p += divisor
    return int(new_p)


def _round_repeats(repeats: int, depth_mult: float) -> int:
    """Scale number of block repeats based on depth multiplier."""
    if not depth_mult:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block with SiLU and Sigmoid activations."""

    in_channels: int
    squeeze_channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # x is [B, H, W, C]
        pooled = jnp.mean(x, axis=(1, 2), keepdims=True)  # [B, 1, 1, C]
        squeezed = nn.Conv(self.squeeze_channels, kernel_size=(1, 1), name="reduce")(pooled)
        squeezed = jax.nn.silu(squeezed)
        excited = nn.Conv(self.in_channels, kernel_size=(1, 1), name="expand")(squeezed)
        excited = jax.nn.sigmoid(excited)
        return x * excited


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block (MBConv) with SE."""

    in_channels: int
    out_channels: int
    expand_ratio: int
    kernel_size: int
    stride: int
    se_ratio: float = 0.25
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        mid_channels = self.in_channels * self.expand_ratio
        res = x

        # 1. Expansion phase (Pointwise Conv)
        if self.expand_ratio != 1:
            x = nn.Conv(mid_channels, kernel_size=(1, 1), use_bias=False, name="expand_conv")(x)
            x = nn.BatchNorm(use_running_average=deterministic, epsilon=1e-5, name="expand_bn")(x)
            x = jax.nn.silu(x)

        # 2. Depthwise Conv with symmetric padding matching PyTorch
        pad = self.kernel_size // 2
        x = nn.Conv(
            mid_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=[(pad, pad), (pad, pad)],
            feature_group_count=mid_channels,
            use_bias=False,
            name="depthwise_conv",
        )(x)
        x = nn.BatchNorm(use_running_average=deterministic, epsilon=1e-5, name="depthwise_bn")(x)
        x = jax.nn.silu(x)

        # 3. Squeeze-and-Excitation
        if self.se_ratio > 0.0:
            squeeze_ch = max(1, int(self.in_channels * self.se_ratio))
            x = SqueezeExcitation(
                in_channels=mid_channels,
                squeeze_channels=squeeze_ch,
                name="se",
            )(x)

        # 4. Pointwise Linear Output Conv
        x = nn.Conv(self.out_channels, kernel_size=(1, 1), use_bias=False, name="project_conv")(x)
        x = nn.BatchNorm(use_running_average=deterministic, epsilon=1e-5, name="project_bn")(x)

        # 5. Residual connection
        if self.stride == 1 and self.in_channels == self.out_channels:
            if not deterministic and self.drop_rate > 0.0:
                x = nn.Dropout(rate=self.drop_rate)(x, deterministic=deterministic)
            x = x + res

        return x


class EfficientNet(nn.Module):
    """
    EfficientNet architecture in Flax Linen.
    Supports feature pyramid extraction (strides 2, 4, 8, 16, 32).
    """

    config: EfficientNetConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        return_features: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, List[jax.Array]]]:
        w_mult = self.config.width_coefficient
        d_mult = self.config.depth_coefficient

        # Base block definitions: (expand_ratio, channels, repeats, stride, kernel_size)
        base_stages = [
            (1, 16, 1, 1, 3),   # Stage 1
            (6, 24, 2, 2, 3),   # Stage 2
            (6, 40, 2, 2, 5),   # Stage 3
            (6, 80, 3, 2, 3),   # Stage 4
            (6, 112, 3, 1, 5),  # Stage 5
            (6, 192, 4, 2, 5),  # Stage 6
            (6, 320, 1, 1, 3),  # Stage 7
        ]

        # 1. Stem Conv with symmetric padding matching PyTorch
        out_stem = _round_channels(32, w_mult)
        x = nn.Conv(
            out_stem,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            name="conv_stem",
        )(x)
        x = nn.BatchNorm(use_running_average=deterministic, epsilon=1e-5, name="bn1")(x)
        x = jax.nn.silu(x)

        in_ch = out_stem
        stage_features: List[jax.Array] = []

        # 2. MBConv Stages
        block_idx = 0
        for stage_i, (expand, ch, repeats, stride, k_size) in enumerate(base_stages):
            out_ch = _round_channels(ch, w_mult)
            num_rep = _round_repeats(repeats, d_mult)

            for rep_i in range(num_rep):
                s = stride if rep_i == 0 else 1
                x = MBConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    expand_ratio=expand,
                    kernel_size=k_size,
                    stride=s,
                    se_ratio=0.25,
                    drop_rate=self.config.drop_connect_rate,
                    name=f"blocks_{block_idx}",
                )(x, deterministic=deterministic)
                in_ch = out_ch
                block_idx += 1

            # Save stage features when downsampled
            if stride == 2 or stage_i == len(base_stages) - 1:
                stage_features.append(x)

        # 3. Head Conv
        out_head = _round_channels(1280, w_mult)
        x = nn.Conv(out_head, kernel_size=(1, 1), use_bias=False, name="conv_head")(x)
        x = nn.BatchNorm(use_running_average=deterministic, epsilon=1e-5, name="bn2")(x)
        x = jax.nn.silu(x)

        # 4. Global Average Pooling & Classifier
        pooled = jnp.mean(x, axis=(1, 2))  # [B, C]
        if not deterministic and self.config.dropout_rate > 0.0:
            pooled = nn.Dropout(rate=self.config.dropout_rate)(pooled, deterministic=deterministic)

        logits = nn.Dense(self.config.num_classes, name="classifier")(pooled)

        if return_features:
            return logits, stage_features
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "efficientnet_b2",
    ) -> Tuple[EfficientNet, Dict[str, Any]]:
        """Load pretrained EfficientNet weights via timm."""
        import timm
        from jax_model_zoo.converters.torch_to_jax import convert_torch_efficientnet_to_jax

        torch_model = timm.create_model(model_name, pretrained=True)
        cfg = EfficientNetConfig(
            width_coefficient=1.1,
            depth_coefficient=1.2,
            image_size=288,
            dropout_rate=0.3,
            num_classes=torch_model.num_classes,
        )
        flax_model = cls(config=cfg)
        jax_params = convert_torch_efficientnet_to_jax(torch_model.state_dict(), cfg)
        return flax_model, jax_params


def efficientnet_b2(pretrained: bool = False, **kwargs) -> Union[EfficientNet, Tuple[EfficientNet, Dict[str, Any]]]:
    """EfficientNet-B2 factory function."""
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet_b2")
    cfg = EfficientNetConfig(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        image_size=288,
        dropout_rate=0.3,
        **kwargs,
    )
    return EfficientNet(config=cfg)
