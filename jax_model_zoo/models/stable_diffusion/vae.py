"""Variational Autoencoder (AutoencoderKL) for Stable Diffusion in Flax Linen (jax.nn)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import AutoencoderKLConfig


class DiagonalGaussianDistribution:
    """Gaussian distribution parameterization for VAE latents."""

    def __init__(self, parameters: jax.Array, deterministic: bool = False):
        # parameters: [B, H, W, 2 * latent_channels]
        self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        self.deterministic = deterministic

    def sample(self, rng: Optional[jax.Array] = None) -> jax.Array:
        if self.deterministic or rng is None:
            return self.mean
        eps = jax.random.normal(rng, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + self.std * eps

    def mode(self) -> jax.Array:
        return self.mean


class ResnetBlock2D(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    in_channels: int
    out_channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        res = x

        h = nn.GroupNorm(num_groups=self.groups, name="norm1")(x)
        h = jax.nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", name="conv1")(h)

        h = nn.GroupNorm(num_groups=self.groups, name="norm2")(h)
        h = jax.nn.silu(h)
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", name="conv2")(h)

        if self.in_channels != self.out_channels:
            res = nn.Conv(self.out_channels, kernel_size=(1, 1), name="conv_shortcut")(res)

        return res + h


class Downsample2D(nn.Module):
    """Downsampling 2x via strided convolution."""

    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="conv",
        )(x)


class Upsample2D(nn.Module):
    """Upsampling 2x via nearest interpolation and convolution."""

    channels: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        b, h, w, c = x.shape
        x_up = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        return nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME", name="conv")(x_up)


class VAEEncoder(nn.Module):
    """Encoder for AutoencoderKL."""

    config: AutoencoderKLConfig

    @nn.compact
    def __call__(self, x: jax.Array, return_skips: bool = False) -> Union[jax.Array, Tuple[jax.Array, List[jax.Array]]]:
        # x is [B, H, W, C]
        ch = self.config.block_out_channels[0]
        x = nn.Conv(ch, kernel_size=(3, 3), padding="SAME", name="conv_in")(x)
        skips = [x]

        curr_ch = ch
        for i, out_ch in enumerate(self.config.block_out_channels):
            for j in range(self.config.layers_per_block):
                x = ResnetBlock2D(
                    in_channels=curr_ch,
                    out_channels=out_ch,
                    groups=self.config.norm_num_groups,
                    name=f"down_blocks_{i}_resnets_{j}",
                )(x)
                curr_ch = out_ch
                skips.append(x)

            if i < len(self.config.block_out_channels) - 1:
                x = Downsample2D(channels=curr_ch, name=f"down_blocks_{i}_downsamplers_0")(x)
                skips.append(x)

        # Mid block
        x = ResnetBlock2D(
            in_channels=curr_ch,
            out_channels=curr_ch,
            groups=self.config.norm_num_groups,
            name="mid_block_resnets_0",
        )(x)
        x = ResnetBlock2D(
            in_channels=curr_ch,
            out_channels=curr_ch,
            groups=self.config.norm_num_groups,
            name="mid_block_resnets_1",
        )(x)

        # Output
        x = nn.GroupNorm(num_groups=self.config.norm_num_groups, name="conv_norm_out")(x)
        x = jax.nn.silu(x)
        # Latent parameters: 2 * latent_channels (mean + logvar)
        x = nn.Conv(2 * self.config.latent_channels, kernel_size=(3, 3), padding="SAME", name="conv_out")(x)

        if return_skips:
            return x, skips
        return x


class VAEDecoder(nn.Module):
    """Decoder for AutoencoderKL."""

    config: AutoencoderKLConfig

    @nn.compact
    def __call__(self, z: jax.Array, return_features: bool = False) -> Union[jax.Array, Tuple[jax.Array, List[jax.Array]]]:
        # z: [B, H/8, W/8, latent_channels]
        reversed_channels = list(reversed(self.config.block_out_channels))
        curr_ch = reversed_channels[0]

        x = nn.Conv(curr_ch, kernel_size=(3, 3), padding="SAME", name="conv_in")(z)

        # Mid block
        x = ResnetBlock2D(
            in_channels=curr_ch,
            out_channels=curr_ch,
            groups=self.config.norm_num_groups,
            name="mid_block_resnets_0",
        )(x)
        x = ResnetBlock2D(
            in_channels=curr_ch,
            out_channels=curr_ch,
            groups=self.config.norm_num_groups,
            name="mid_block_resnets_1",
        )(x)

        stage_features = [x]

        # Up blocks
        for i, out_ch in enumerate(reversed_channels):
            for j in range(self.config.layers_per_block + 1):
                x = ResnetBlock2D(
                    in_channels=curr_ch,
                    out_channels=out_ch,
                    groups=self.config.norm_num_groups,
                    name=f"up_blocks_{i}_resnets_{j}",
                )(x)
                curr_ch = out_ch

            if i < len(reversed_channels) - 1:
                x = Upsample2D(channels=curr_ch, name=f"up_blocks_{i}_upsamplers_0")(x)
                stage_features.append(x)

        # Output projection
        x = nn.GroupNorm(num_groups=self.config.norm_num_groups, name="conv_norm_out")(x)
        x = jax.nn.silu(x)
        out = nn.Conv(self.config.out_channels, kernel_size=(3, 3), padding="SAME", name="conv_out")(x)
        stage_features.append(out)

        if return_features:
            return out, stage_features
        return out


class AutoencoderKL(nn.Module):
    """Complete AutoencoderKL model in Flax Linen."""

    config: AutoencoderKLConfig

    def setup(self):
        self.encoder = VAEEncoder(config=self.config)
        self.quant_conv = nn.Conv(2 * self.config.latent_channels, kernel_size=(1, 1), name="quant_conv")
        self.post_quant_conv = nn.Conv(self.config.latent_channels, kernel_size=(1, 1), name="post_quant_conv")
        self.decoder = VAEDecoder(config=self.config)

    def __call__(
        self,
        x: jax.Array,
        sample_posterior: bool = False,
        rng: Optional[jax.Array] = None,
    ) -> jax.Array:
        # Encode
        posterior = self.encode(x)
        z = posterior.sample(rng) if sample_posterior else posterior.mode()
        # Decode
        return self.decode(z)

    def encode(self, x: jax.Array, return_skips: bool = False) -> Union[DiagonalGaussianDistribution, Tuple[DiagonalGaussianDistribution, List[jax.Array]]]:
        if return_skips:
            params, skips = self.encoder(x, return_skips=True)
            quant = self.quant_conv(params)
            return DiagonalGaussianDistribution(quant), skips
        params = self.encoder(x, return_skips=False)
        quant = self.quant_conv(params)
        return DiagonalGaussianDistribution(quant)

    def decode(self, z: jax.Array, return_features: bool = False) -> Union[jax.Array, Tuple[jax.Array, List[jax.Array]]]:
        z_post = self.post_quant_conv(z)
        return self.decoder(z_post, return_features=return_features)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        subfolder: str = "vae",
    ) -> Tuple[AutoencoderKL, Dict[str, Any]]:
        """Load pretrained VAE from HuggingFace diffusers."""
        from diffusers import AutoencoderKL as HFAutoencoderKL
        from jax_model_zoo.converters.torch_to_jax import convert_torch_vae_to_jax

        hf_vae = HFAutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
        cfg = AutoencoderKLConfig(
            block_out_channels=tuple(hf_vae.config.block_out_channels),
            latent_channels=hf_vae.config.latent_channels,
            norm_num_groups=hf_vae.config.norm_num_groups,
            scaling_factor=hf_vae.config.scaling_factor,
        )
        flax_model = cls(config=cfg)
        jax_params = convert_torch_vae_to_jax(hf_vae.state_dict(), cfg)
        return flax_model, jax_params
