"""Tests for EfficientNet-B2 backbone."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.config import EfficientNetConfig
from jax_model_zoo.converters.torch_to_jax import convert_torch_efficientnet_to_jax
from jax_model_zoo.models.efficientnet.efficientnet import (
    EfficientNet,
    efficientnet_b2,
)


def test_efficientnet_b2_forward_shape():
    model = efficientnet_b2(num_classes=10)
    # Using 64x64 with jax.jit for rapid unit test execution
    x = jnp.ones((2, 64, 64, 3))
    rng = jax.random.PRNGKey(0)

    variables = model.init(rng, x)
    apply_fn = jax.jit(model.apply)
    logits = apply_fn(variables, x)
    assert logits.shape == (2, 10)


def test_efficientnet_feature_pyramid():
    cfg = EfficientNetConfig(width_coefficient=0.5, depth_coefficient=0.5, num_classes=10)
    model = EfficientNet(config=cfg)
    x = jnp.ones((1, 64, 64, 3))
    rng = jax.random.PRNGKey(42)

    variables = model.init(rng, x)
    apply_fn = jax.jit(lambda v, inp: model.apply(v, inp, return_features=True))
    logits, stage_features = apply_fn(variables, x)

    assert logits.shape == (1, 10)
    assert len(stage_features) >= 4
    # Check that spatial dimensions decrease across stages
    assert stage_features[0].shape[1] > stage_features[-1].shape[1]


def test_efficientnet_torch_conversion():
    import torch

    torch_state = {
        "conv_stem.weight": torch.randn(32, 3, 3, 3),
        "bn1.weight": torch.ones(32),
        "bn1.bias": torch.zeros(32),
        "bn1.running_mean": torch.zeros(32),
        "bn1.running_var": torch.ones(32),
        "conv_head.weight": torch.randn(1280, 320, 1, 1),
        "bn2.weight": torch.ones(1280),
        "bn2.bias": torch.zeros(1280),
        "classifier.weight": torch.randn(10, 1280),
        "classifier.bias": torch.zeros(10),
    }

    cfg = EfficientNetConfig(num_classes=10)
    jax_params = convert_torch_efficientnet_to_jax(torch_state, cfg)

    assert "params" in jax_params
    assert "conv_stem" in jax_params["params"]
    assert jax_params["params"]["conv_stem"]["kernel"].shape == (3, 3, 3, 32)
    assert jax_params["params"]["classifier"]["kernel"].shape == (1280, 10)
