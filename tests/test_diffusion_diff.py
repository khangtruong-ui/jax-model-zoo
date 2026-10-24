"""Tests for DiffusionDiff models family (DiffusionDiff, DiffusionDiffV2, DiffusionDiffMinimized)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.config import DiffusionDiffConfig
from jax_model_zoo.converters.jax_to_torch import convert_jax_diffusion_diff_to_torch
from jax_model_zoo.converters.torch_to_jax import convert_torch_diffusion_diff_to_jax
from jax_model_zoo.models.diffusion_diff.blocks import (
    expand_to_spatial,
    sinusoidal_embedding,
)
from jax_model_zoo.models.diffusion_diff.diffusion_diff import DiffusionDiff
from jax_model_zoo.models.diffusion_diff.diffusion_diff_minimized import DiffusionDiffMinimized
from jax_model_zoo.models.diffusion_diff.diffusion_diff_v2 import DiffusionDiffV2


def test_sinusoidal_embedding():
    timesteps = jnp.array([10.0, 100.0, 500.0])
    emb = sinusoidal_embedding(timesteps, dim=32)
    assert emb.shape == (3, 32)
    # Different inputs yield distinct embeddings
    assert not jnp.allclose(emb[0], emb[1])

    spatial = expand_to_spatial(emb, height=16, width=16)
    assert spatial.shape == (3, 16, 16, 32)
    np.testing.assert_allclose(np.asarray(spatial[:, 0, 0, :]), np.asarray(emb))


def test_diffusion_diff_shapes_with_and_without_aux_classifier():
    # Use dummy lightweight backbone for fast unit testing
    cfg_with_aux = DiffusionDiffConfig(
        timesteps=[100, 250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_channels=[32, 16, 8],
        aux_classifier=True,
        num_classes=3,
        is_efficient_backbone=True,
    )
    model = DiffusionDiff(config=cfg_with_aux)
    x = jnp.ones((2, 64, 64, 3))
    rng = jax.random.PRNGKey(0)

    params = model.init(rng, x, rng=rng)
    jitted_forward = jax.jit(lambda p, v: model.apply(p, v, rng=rng))
    mask_logits, class_logits = jitted_forward(params, x)

    assert mask_logits.shape == (2, 64, 64, 1)
    assert class_logits.shape == (2, 3)

    # Without aux classifier
    cfg_no_aux = DiffusionDiffConfig(
        timesteps=[100],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_channels=[32, 16, 8],
        aux_classifier=False,
        is_efficient_backbone=True,
    )
    model2 = DiffusionDiff(config=cfg_no_aux)
    params2 = model2.init(rng, x, rng=rng)
    jitted_forward2 = jax.jit(lambda p, v: model2.apply(p, v, rng=rng))
    mask_logits2 = jitted_forward2(params2, x)
    assert mask_logits2.shape == (2, 64, 64, 1)


def test_diffusion_diff_v2_shapes():
    cfg = DiffusionDiffConfig(
        version="v2",
        timesteps=[100, 500],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_channels=[32, 16, 8],
        aux_classifier=True,
        num_classes=2,
        is_efficient_backbone=True,
    )
    model = DiffusionDiffV2(config=cfg)
    x = jnp.ones((2, 64, 64, 3))
    rng = jax.random.PRNGKey(42)

    params = model.init(rng, x, rng=rng)
    jitted_forward = jax.jit(lambda p, v: model.apply(p, v, rng=rng))
    mask_logits, class_logits = jitted_forward(params, x)

    assert mask_logits.shape == (2, 64, 64, 1)
    assert class_logits.shape == (2, 2)


def test_diffusion_diff_minimized_two_lines_of_computation():
    """
    Verify Diffusion-Diff-Minimized strictly adheres to:
    1. Line 1: Real image clean latent z0 (4 channels)
    2. Line 2: Single noisy latent (80 channels: 4 noisy + 4 eps + 4 pred + 4 diff + 32 t_emb + 32 sigma_emb)
    Total Z = 84 channels.
    """
    cfg = DiffusionDiffConfig(
        version="minimized",
        timesteps=[250],
        timestep_embed_dim=32,
        sigma_embed_dim=32,
        decoder_channels=[32, 16, 8],
        is_efficient_backbone=True,
    )
    model = DiffusionDiffMinimized(config=cfg)

    assert len(model.config.timesteps) == 1
    assert model.config.timesteps == [250]
    assert model.total_z_channels == 84

    x = jnp.ones((1, 64, 64, 3))
    rng = jax.random.PRNGKey(7)

    params = model.init(rng, x, rng=rng)
    jitted_forward = jax.jit(lambda p, v: model.apply(p, v, rng=rng))
    mask_logits = jitted_forward(params, x)
    assert mask_logits.shape == (1, 64, 64, 1)


def test_diffusion_diff_checkpoint_signature_parity():
    """Verify state dict and checkpoint keys match between PyTorch and JAX versions."""
    import torch

    torch_state = {
        "z_norm.weight": torch.ones(84),
        "z_norm.bias": torch.zeros(84),
        "decoder.init_conv.0.weight": torch.randn(32, 84, 3, 3),
        "decoder.init_conv.1.weight": torch.ones(32),
        "decoder.init_conv.1.bias": torch.zeros(32),
        "decoder.stages.0.up.1.weight": torch.randn(16, 32, 3, 3),
        "decoder.stages.0.norm.weight": torch.ones(16),
        "decoder.stages.0.norm.bias": torch.zeros(16),
        "decoder.stages.0.res_blocks.0.conv1.weight": torch.randn(16, 16, 3, 3),
        "decoder.stages.0.res_blocks.0.norm1.weight": torch.ones(16),
        "decoder.stages.0.res_blocks.0.norm1.bias": torch.zeros(16),
        "decoder.stages.0.res_blocks.0.conv2.weight": torch.randn(16, 16, 3, 3),
        "decoder.stages.0.res_blocks.0.norm2.weight": torch.ones(16),
        "decoder.stages.0.res_blocks.0.norm2.bias": torch.zeros(16),
        "decoder.out_conv.weight": torch.randn(1, 16, 3, 3),
        "decoder.out_conv.bias": torch.zeros(1),
        "classifier_head.fc1.weight": torch.randn(128, 84),
        "classifier_head.fc1.bias": torch.zeros(128),
        "classifier_head.fc2.weight": torch.randn(2, 128),
        "classifier_head.fc2.bias": torch.zeros(2),
    }

    cfg = DiffusionDiffConfig(decoder_channels=[32, 16], aux_classifier=True)
    jax_params = convert_torch_diffusion_diff_to_jax(torch_state, cfg)

    assert "params" in jax_params
    assert "z_norm" in jax_params["params"]
    assert "decoder" in jax_params["params"]
    assert "classifier_head" in jax_params["params"]

    # Convert back to torch and check signature preservation
    torch_back = convert_jax_diffusion_diff_to_torch(jax_params, cfg)
    for k in [
        "z_norm.weight",
        "decoder.init_conv.0.weight",
        "decoder.stages.0.up.1.weight",
        "decoder.out_conv.weight",
        "classifier_head.fc1.weight",
    ]:
        assert k in torch_back, f"Missing key {k} in converted torch state"
        assert torch_back[k].shape == torch_state[k].shape
