"""UNet2DConditionModel for Stable Diffusion in Flax Linen (jax.nn)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import UNet2DConditionConfig


def sinusoidal_embedding(
    timesteps: jax.Array,
    dim: int,
    max_period: float = 10000.0,
) -> jax.Array:
    """Compute sinusoidal positional embeddings for scalar timesteps."""
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
    return embedding


class TimestepEmbedding(nn.Module):
    """MLP projecting sinusoidal timestep embeddings to conditioning dimension."""

    time_embed_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.time_embed_dim, name="linear_1")(x)
        x = jax.nn.silu(x)
        x = nn.Dense(self.time_embed_dim, name="linear_2")(x)
        return x


class UNetResnetBlock(nn.Module):
    """ResNet block conditioned on timestep embedding."""

    in_channels: int
    out_channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x: jax.Array, time_emb: Optional[jax.Array] = None) -> jax.Array:
        res = x
        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6, name="norm1")(x)
        h = jax.nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name="conv1")(h)

        if time_emb is not None:
            time_proj = nn.Dense(self.out_channels, name="time_emb_proj")(jax.nn.silu(time_emb))
            h = h + time_proj[:, None, None, :]

        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6, name="norm2")(h)
        h = jax.nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name="conv2")(h)

        if self.in_channels != self.out_channels:
            res = nn.Conv(self.out_channels, kernel_size=(1, 1), name="conv_shortcut")(res)

        return res + h


class CrossAttention(nn.Module):
    """Multi-Head Cross/Self Attention block."""

    query_dim: int
    context_dim: Optional[int] = None
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        # x is [B, N, query_dim]
        b, n, _ = x.shape
        inner_dim = self.heads * self.dim_head
        ctx = x if context is None else context

        q = nn.Dense(inner_dim, use_bias=False, name="to_q")(x)
        k = nn.Dense(inner_dim, use_bias=False, name="to_k")(ctx)
        v = nn.Dense(inner_dim, use_bias=False, name="to_v")(ctx)

        q = q.reshape((b, -1, self.heads, self.dim_head))
        k = k.reshape((b, -1, self.heads, self.dim_head))
        v = v.reshape((b, -1, self.heads, self.dim_head))

        q = jnp.swapaxes(q, 1, 2)  # [B, heads, N_q, dim_head]
        k = jnp.swapaxes(k, 1, 2)  # [B, heads, N_k, dim_head]
        v = jnp.swapaxes(v, 1, 2)  # [B, heads, N_k, dim_head]

        scale = jnp.array(1.0 / np.sqrt(self.dim_head), dtype=q.dtype)
        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        weights = jax.nn.softmax(scores, axis=-1)

        out = jnp.matmul(weights, v)  # [B, heads, N_q, dim_head]
        out = jnp.swapaxes(out, 1, 2).reshape((b, n, inner_dim))
        return nn.Dense(self.query_dim, name="to_out_0")(out)


class Transformer2DModel(nn.Module):
    """Spatial Transformer Block with Self-Attention, Cross-Attention, and FeedForward."""

    dim: int
    n_heads: int = 8
    d_head: int = 64
    context_dim: Optional[int] = 768
    norm_num_groups: int = 32

    @nn.compact
    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        # x is [B, H, W, C]
        b, h, w, c = x.shape
        res = x
        x_norm = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6, name="norm")(x)
        x_flat = nn.Conv(c, kernel_size=(1, 1), name="proj_in")(x_norm).reshape((b, h * w, c))

        # 1. Self-Attention
        attn1_in = nn.LayerNorm(epsilon=1e-5, name="norm1")(x_flat)
        x_flat = x_flat + CrossAttention(
            query_dim=c,
            heads=self.n_heads,
            dim_head=self.d_head,
            name="attn1",
        )(attn1_in, None)

        # 2. Cross-Attention
        if context is not None:
            attn2_in = nn.LayerNorm(epsilon=1e-5, name="norm2")(x_flat)
            x_flat = x_flat + CrossAttention(
                query_dim=c,
                context_dim=self.context_dim,
                heads=self.n_heads,
                dim_head=self.d_head,
                name="attn2",
            )(attn2_in, context)

        # 3. FeedForward (GEGLU MLP)
        ff_in = nn.LayerNorm(epsilon=1e-5, name="norm3")(x_flat)
        h_geglu = nn.Dense(c * 4 * 2, name="ff_net_0_proj")(ff_in)
        h1, h2 = jnp.split(h_geglu, 2, axis=-1)
        h_act = h1 * jax.nn.gelu(h2, approximate=False)
        h_out = nn.Dense(c, name="ff_net_2")(h_act)
        x_flat = x_flat + h_out

        x_spatial = x_flat.reshape((b, h, w, c))
        out = nn.Conv(c, kernel_size=(1, 1), name="proj_out")(x_spatial)
        return res + out


class UNetDownsample2D(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(self.channels, kernel_size=(3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)], name="conv")(x)


class UNetUpsample2D(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        b, h, w, c = x.shape
        x_up = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        return nn.Conv(self.channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name="conv")(x_up)


class UNet2DConditionModel(nn.Module):
    """UNet2DConditionModel backbone in Flax Linen."""

    config: UNet2DConditionConfig

    @nn.compact
    def __call__(
        self,
        sample: jax.Array,
        timestep: jax.Array,
        encoder_hidden_states: Optional[jax.Array] = None,
    ) -> jax.Array:
        # sample: [B, H, W, 4]
        b = sample.shape[0]
        time_dim = self.config.block_out_channels[0] * 4

        # 1. Timestep embedding
        t_freq = sinusoidal_embedding(timestep, dim=self.config.block_out_channels[0])
        time_emb = TimestepEmbedding(time_embed_dim=time_dim, name="time_embedding")(t_freq)

        # 2. Stem conv
        curr_ch = self.config.block_out_channels[0]
        x = nn.Conv(curr_ch, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name="conv_in")(sample)
        down_block_res_samples = [x]

        # 3. Down Blocks
        for i, out_ch in enumerate(self.config.block_out_channels):
            type_idx = min(i, len(self.config.down_block_types) - 1)
            is_cross = "CrossAttn" in self.config.down_block_types[type_idx]
            for j in range(self.config.layers_per_block):
                x = UNetResnetBlock(
                    in_channels=curr_ch,
                    out_channels=out_ch,
                    groups=self.config.norm_num_groups,
                    name=f"down_blocks_{i}_resnets_{j}",
                )(x, time_emb=time_emb)
                curr_ch = out_ch

                if is_cross:
                    heads = 8
                    dim_head = max(1, out_ch // heads)
                    x = Transformer2DModel(
                        dim=out_ch,
                        n_heads=heads,
                        d_head=dim_head,
                        context_dim=self.config.cross_attention_dim,
                        norm_num_groups=self.config.norm_num_groups,
                        name=f"down_blocks_{i}_attentions_{j}",
                    )(x, context=encoder_hidden_states)
                down_block_res_samples.append(x)

            if i < len(self.config.block_out_channels) - 1:
                x = UNetDownsample2D(channels=curr_ch, name=f"down_blocks_{i}_downsamplers_0")(x)
                down_block_res_samples.append(x)

        # 4. Mid Block
        mid_ch = self.config.block_out_channels[-1]
        x = UNetResnetBlock(
            in_channels=mid_ch,
            out_channels=mid_ch,
            groups=self.config.norm_num_groups,
            name="mid_block_resnets_0",
        )(x, time_emb=time_emb)
        x = Transformer2DModel(
            dim=mid_ch,
            n_heads=8,
            d_head=mid_ch // 8,
            context_dim=self.config.cross_attention_dim,
            norm_num_groups=self.config.norm_num_groups,
            name="mid_block_attentions_0",
        )(x, context=encoder_hidden_states)
        x = UNetResnetBlock(
            in_channels=mid_ch,
            out_channels=mid_ch,
            groups=self.config.norm_num_groups,
            name="mid_block_resnets_1",
        )(x, time_emb=time_emb)

        # 5. Up Blocks
        reversed_channels = list(reversed(self.config.block_out_channels))
        up_types = list(self.config.up_block_types)

        for i, out_ch in enumerate(reversed_channels):
            type_idx = min(i, len(up_types) - 1)
            is_cross = "CrossAttn" in up_types[type_idx]
            for j in range(self.config.layers_per_block + 1):
                skip = down_block_res_samples.pop()
                x = jnp.concatenate([x, skip], axis=-1)
                in_ch = curr_ch + skip.shape[-1]
                x = UNetResnetBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    groups=self.config.norm_num_groups,
                    name=f"up_blocks_{i}_resnets_{j}",
                )(x, time_emb=time_emb)
                curr_ch = out_ch

                if is_cross:
                    heads = 8
                    dim_head = max(1, out_ch // heads)
                    x = Transformer2DModel(
                        dim=out_ch,
                        n_heads=heads,
                        d_head=dim_head,
                        context_dim=self.config.cross_attention_dim,
                        norm_num_groups=self.config.norm_num_groups,
                        name=f"up_blocks_{i}_attentions_{j}",
                    )(x, context=encoder_hidden_states)

            if i < len(reversed_channels) - 1:
                x = UNetUpsample2D(channels=curr_ch, name=f"up_blocks_{i}_upsamplers_0")(x)

        # 6. Final Conv
        x = nn.GroupNorm(num_groups=self.config.norm_num_groups, epsilon=1e-5, name="conv_norm_out")(x)
        x = jax.nn.silu(x)
        out = nn.Conv(self.config.out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name="conv_out")(x)
        return out
        return out

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        subfolder: str = "unet",
    ) -> Tuple[UNet2DConditionModel, Dict[str, Any]]:
        """Load pretrained UNet from HuggingFace diffusers."""
        from diffusers import UNet2DConditionModel as HFUNet
        from jax_model_zoo.converters.torch_to_jax import convert_torch_unet_to_jax

        hf_unet = HFUNet.from_pretrained(model_name, subfolder=subfolder)
        cfg = UNet2DConditionConfig(
            block_out_channels=tuple(hf_unet.config.block_out_channels),
            down_block_types=tuple(hf_unet.config.down_block_types),
            up_block_types=tuple(hf_unet.config.up_block_types),
            cross_attention_dim=hf_unet.config.cross_attention_dim,
            norm_num_groups=hf_unet.config.norm_num_groups,
        )
        flax_model = cls(config=cfg)
        jax_params = convert_torch_unet_to_jax(hf_unet.state_dict(), cfg)
        return flax_model, jax_params
