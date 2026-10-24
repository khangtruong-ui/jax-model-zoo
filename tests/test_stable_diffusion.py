"""Tests for Stable Diffusion components (VAE, UNet, and Pipelines)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.config import AutoencoderKLConfig, UNet2DConditionConfig
from jax_model_zoo.models.stable_diffusion.pipeline import (
    stable_diffusion_efficient,
    stable_diffusion_v1_5,
)
from jax_model_zoo.models.stable_diffusion.unet import UNet2DConditionModel
from jax_model_zoo.models.stable_diffusion.vae import AutoencoderKL


def test_vae_encode_decode():
    # Lightweight dummy VAE for fast testing
    cfg = AutoencoderKLConfig(
        block_out_channels=(16, 32),
        latent_channels=4,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(config=cfg)
    x = jnp.ones((2, 32, 32, 3))
    rng = jax.random.PRNGKey(0)

    params = vae.init(rng, x)
    out = vae.apply(params, x)
    assert out.shape == (2, 32, 32, 3)

    # Test posterior mode and sampling
    posterior = vae.apply(params, x, method=vae.encode)
    latents = posterior.mode()
    # 1 downsampling stage between 2 blocks: 32 -> 16
    assert latents.shape == (2, 16, 16, 4)

    recon = vae.apply(params, latents, method=vae.decode)
    assert recon.shape == (2, 32, 32, 3)


def test_unet_condition_forward():
    # Lightweight dummy UNet for fast testing
    cfg = UNet2DConditionConfig(
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        norm_num_groups=8,
    )
    unet = UNet2DConditionModel(config=cfg)
    sample = jnp.ones((2, 16, 16, 4))
    timestep = jnp.array([100, 500])
    encoder_hidden_states = jnp.ones((2, 4, 32))
    rng = jax.random.PRNGKey(42)

    params = unet.init(rng, sample, timestep, encoder_hidden_states)
    pred = unet.apply(params, sample, timestep, encoder_hidden_states)

    assert pred.shape == (2, 16, 16, 4)


def test_sd_pipeline_factory():
    pipe_v15 = stable_diffusion_v1_5()
    assert pipe_v15.config.is_efficient is False
    assert len(pipe_v15.config.unet.block_out_channels) == 4

    pipe_eff = stable_diffusion_efficient()
    assert pipe_eff.config.is_efficient is True
    # Efficient / Tiny-SD has 3 stages (320, 640, 1280)
    assert len(pipe_eff.config.unet.block_out_channels) == 3
