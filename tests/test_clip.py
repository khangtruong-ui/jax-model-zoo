"""Tests for CLIP (Contrastive Language-Image Pre-Training)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from jax_model_zoo.converters.torch_to_jax import convert_torch_clip_to_jax
from jax_model_zoo.models.clip.clip import (
    CLIP,
    CLIPTextTransformer,
    CLIPVisionTransformer,
    clip_vit_base_patch32,
)


def test_clip_vit_base_patch32_shapes():
    cfg = CLIPConfig(
        vision=CLIPVisionConfig(image_size=64, patch_size=16, width=64, layers=2, heads=2, output_dim=32),
        text=CLIPTextConfig(context_length=16, vocab_size=1000, width=64, heads=2, layers=2, output_dim=32),
    )
    model = CLIP(config=cfg)

    pixel_values = jnp.ones((2, 64, 64, 3))
    input_ids = jnp.ones((3, 16), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)

    params = model.init(rng, pixel_values, input_ids)

    # Encode image and text separately
    img_emb = model.apply(params, pixel_values, method=model.encode_image)
    assert img_emb.shape == (2, 32)
    # Check normalization
    norms = jnp.linalg.norm(img_emb, axis=-1)
    np.testing.assert_allclose(np.asarray(norms), np.ones(2), rtol=1e-4)

    text_emb = model.apply(params, input_ids, method=model.encode_text)
    assert text_emb.shape == (3, 32)

    # Joint contrastive forward pass
    logits_per_image, logits_per_text = model.apply(params, pixel_values, input_ids)
    assert logits_per_image.shape == (2, 3)
    assert logits_per_text.shape == (3, 2)


def test_clip_torch_conversion():
    import torch

    torch_state = {
        "logit_scale": torch.tensor(2.6592),
        "visual.conv1.weight": torch.randn(64, 3, 16, 16),
        "visual.class_embedding": torch.randn(64),
        "visual.positional_embedding": torch.randn(17, 64),
        "visual.ln_pre.weight": torch.ones(64),
        "visual.ln_pre.bias": torch.zeros(64),
        "visual.ln_post.weight": torch.ones(64),
        "visual.ln_post.bias": torch.zeros(64),
        "visual.proj": torch.randn(64, 32),
        "token_embedding.weight": torch.randn(1000, 64),
        "text.positional_embedding": torch.randn(16, 64),
        "text_projection": torch.randn(64, 32),
        "ln_final.weight": torch.ones(64),
        "ln_final.bias": torch.zeros(64),
    }

    cfg = CLIPConfig()
    jax_params = convert_torch_clip_to_jax(torch_state, cfg)

    assert "params" in jax_params
    assert "visual" in jax_params["params"]
    assert "text" in jax_params["params"]
    assert jax_params["params"]["visual"]["conv1"]["kernel"].shape == (16, 16, 3, 64)
    assert jax_params["params"]["visual"]["proj"]["kernel"].shape == (64, 32)
