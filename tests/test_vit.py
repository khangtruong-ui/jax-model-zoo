"""Tests for Vision Transformer (ViT-B, ViT-L)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.config import ViTConfig
from jax_model_zoo.converters.torch_to_jax import convert_torch_vit_to_jax
from jax_model_zoo.converters.jax_to_torch import convert_jax_vit_to_torch
from jax_model_zoo.models.vision_transformer.vit import (
    VisionTransformer,
    vit_base_patch16_224,
    vit_large_patch14_224,
    vit_large_patch16_224,
)


def test_vit_base_forward_shape():
    model = vit_base_patch16_224(num_classes=10)
    x = jnp.ones((2, 224, 224, 3))
    rng = jax.random.PRNGKey(0)

    params = model.init(rng, x)
    logits = model.apply(params, x)
    assert logits.shape == (2, 10)


def test_vit_large_forward_and_feature_extraction():
    # Use lightweight config with patch_size=16 and depth=2 for fast test
    cfg = ViTConfig(
        img_size=64,
        patch_size=16,
        embed_dim=128,
        depth=2,
        num_heads=4,
        num_classes=5,
    )
    model = VisionTransformer(config=cfg)
    x = jnp.ones((1, 64, 64, 3))
    rng = jax.random.PRNGKey(42)

    params = model.init(rng, x)
    logits, intermediate_features = model.apply(params, x, return_features=True)

    assert logits.shape == (1, 5)
    assert len(intermediate_features) == 2
    # Number of patches = (64/16)^2 = 16 patches + 1 cls token = 17 tokens
    assert intermediate_features[0].shape == (1, 17, 128)


def test_vit_torch_conversion_parity():
    import torch
    import torch.nn as nn

    # Create dummy torch state dict for ViT
    dim = 64
    torch_state = {
        "vit.embeddings.patch_embeddings.projection.weight": torch.randn(dim, 3, 16, 16),
        "vit.embeddings.patch_embeddings.projection.bias": torch.randn(dim),
        "vit.embeddings.cls_token": torch.randn(1, 1, dim),
        "vit.embeddings.position_embeddings": torch.randn(1, 17, dim),
        "vit.layernorm.weight": torch.ones(dim),
        "vit.layernorm.bias": torch.zeros(dim),
        "classifier.weight": torch.randn(5, dim),
        "classifier.bias": torch.randn(5),
    }

    cfg = ViTConfig(img_size=64, patch_size=16, embed_dim=dim, depth=1, num_heads=2, num_classes=5)
    jax_params = convert_torch_vit_to_jax(torch_state, cfg)

    assert "params" in jax_params
    assert "patch_embed" in jax_params["params"]
    assert jax_params["params"]["patch_embed"]["projection"]["kernel"].shape == (16, 16, 3, dim)
    assert jax_params["params"]["head"]["kernel"].shape == (dim, 5)

    # Test reverse conversion back to torch
    torch_converted = convert_jax_vit_to_torch(jax_params, cfg)
    assert "vit.embeddings.patch_embeddings.projection.weight" in torch_converted
    assert torch_converted["vit.embeddings.patch_embeddings.projection.weight"].shape == (dim, 3, 16, 16)
