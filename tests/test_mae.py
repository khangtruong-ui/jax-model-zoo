"""Tests for Masked Autoencoder (MAE)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.config import MAEConfig
from jax_model_zoo.models.vision_transformer.mae import (
    MaskedAutoencoder,
    mae_vit_base_patch16,
    random_masking,
)


def test_random_masking():
    b, n, d = 2, 16, 32
    x = jnp.ones((b, n, d))
    rng = jax.random.PRNGKey(0)

    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75, rng=rng)

    # 75% masked = 25% kept = 4 kept patches
    assert x_masked.shape == (2, 4, 32)
    assert mask.shape == (2, 16)
    assert ids_restore.shape == (2, 16)
    # 4 zeros (kept), 12 ones (masked)
    assert jnp.sum(mask[0]) == 12


def test_mae_forward_reconstruction():
    cfg = MAEConfig(
        img_size=64,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        mask_ratio=0.75,
    )
    model = MaskedAutoencoder(config=cfg)
    x = jnp.ones((2, 64, 64, 3))
    rng = jax.random.PRNGKey(42)

    params = model.init(rng, x, rng=rng)
    pred, mask = model.apply(params, x, rng=rng)

    num_patches = (64 // 16) ** 2  # 16 patches
    patch_dim = 16 * 16 * 3        # 768 pixels per patch
    assert pred.shape == (2, num_patches, patch_dim)
    assert mask.shape == (2, num_patches)


def test_mae_extract_features():
    cfg = MAEConfig(
        img_size=64,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
    )
    model = MaskedAutoencoder(config=cfg)
    x = jnp.ones((1, 64, 64, 3))
    rng = jax.random.PRNGKey(10)

    params = model.init(rng, x)
    features = model.apply(params, x, method=model.extract_features)

    # 16 patches + 1 cls token = 17 tokens
    assert features.shape == (1, 17, 64)
