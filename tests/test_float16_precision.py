"""Tests for float16 and bfloat16 lower precision stability across jax_model_zoo."""

import pytest
import jax
import jax.numpy as jnp
from jax_model_zoo.models.clip.clip import CLIPAttention
from jax_model_zoo.models.vision_transformer.vit import ViTAttention
from jax_model_zoo.models.stable_diffusion.unet import CrossAttention as UNetAttention
from jax_model_zoo.models.stable_diffusion.vae import MidBlockAttention
from jax_model_zoo.models.diffusion_diff.diffusion_diff_minimized import DiffusionDiffMinimized
from jax_model_zoo.core.config import DiffusionDiffConfig


@pytest.mark.timeout(60)
def test_clip_attention_float16_dtype_preservation():
    mod = CLIPAttention(d_model=64, n_head=4)
    x = jnp.ones((2, 8, 64), dtype=jnp.float16)
    vars = mod.init(jax.random.PRNGKey(0), x)
    vars_f16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.float16) if jnp.issubdtype(p.dtype, jnp.floating) else p, vars)
    out = mod.apply(vars_f16, x)
    assert out.dtype == jnp.float16, f"Expected float16, got {out.dtype}"


@pytest.mark.timeout(60)
def test_vit_attention_float16_dtype_preservation():
    mod = ViTAttention(dim=64, num_heads=4)
    x = jnp.ones((2, 16, 64), dtype=jnp.float16)
    vars = mod.init(jax.random.PRNGKey(0), x)
    vars_f16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.float16) if jnp.issubdtype(p.dtype, jnp.floating) else p, vars)
    out = mod.apply(vars_f16, x, deterministic=True)
    assert out.dtype == jnp.float16, f"Expected float16, got {out.dtype}"


@pytest.mark.timeout(60)
def test_unet_attention_float16_dtype_preservation():
    mod = UNetAttention(query_dim=64, heads=4, dim_head=16)
    x = jnp.ones((2, 16, 64), dtype=jnp.float16)
    vars = mod.init(jax.random.PRNGKey(0), x)
    vars_f16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.float16) if jnp.issubdtype(p.dtype, jnp.floating) else p, vars)
    out = mod.apply(vars_f16, x)
    assert out.dtype == jnp.float16, f"Expected float16, got {out.dtype}"


@pytest.mark.timeout(60)
def test_vae_mid_block_attention_float16_dtype_preservation():
    mod = MidBlockAttention(channels=32, groups=8)
    x = jnp.ones((2, 4, 4, 32), dtype=jnp.float16)
    vars = mod.init(jax.random.PRNGKey(0), x)
    vars_f16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.float16) if jnp.issubdtype(p.dtype, jnp.floating) else p, vars)
    out = mod.apply(vars_f16, x)
    assert out.dtype == jnp.float16, f"Expected float16, got {out.dtype}"


@pytest.mark.timeout(60)
def test_bfloat16_precision_stability():
    mod = CLIPAttention(d_model=64, n_head=4)
    x = jnp.ones((2, 8, 64), dtype=jnp.bfloat16)
    vars = mod.init(jax.random.PRNGKey(0), x)
    vars_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if jnp.issubdtype(p.dtype, jnp.floating) else p, vars)
    out = mod.apply(vars_bf16, x)
    assert out.dtype == jnp.bfloat16, f"Expected bfloat16, got {out.dtype}"
