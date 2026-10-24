"""Tests for model surgery utilities: LoRA, head replacement, and freezing."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.surgery import (
    LoRAConv,
    LoRADense,
    get_trainable_param_mask,
    inject_lora,
    merge_lora,
    replace_head,
)


def test_lora_dense():
    layer = LoRADense(features=16, rank=4, alpha=2.0)
    x = jnp.ones((2, 8))
    rng = jax.random.PRNGKey(0)

    params = layer.init(rng, x)
    assert "lora_a" in params["params"]
    assert "lora_b" in params["params"]
    assert params["params"]["lora_a"].shape == (8, 4)
    assert params["params"]["lora_b"].shape == (4, 16)

    out = layer.apply(params, x)
    assert out.shape == (2, 16)


def test_lora_conv():
    conv = LoRAConv(features=8, kernel_size=(3, 3), rank=2, alpha=1.0)
    x = jnp.ones((2, 16, 16, 4))
    rng = jax.random.PRNGKey(0)

    params = conv.init(rng, x)
    assert "lora_a" in params["params"]
    assert "lora_b" in params["params"]

    out = conv.apply(params, x)
    assert out.shape == (2, 16, 16, 8)


def test_inject_and_merge_lora():
    params = {
        "params": {
            "encoder": {
                "q_proj": {"kernel": jnp.ones((16, 16))},
                "other": {"kernel": jnp.ones((16, 8))},
            }
        }
    }

    injected = inject_lora(params, target_patterns=["q_proj"], rank=4, alpha=1.0)
    q_params = injected["params"]["encoder"]["q_proj"]
    assert "lora_a" in q_params
    assert "lora_b" in q_params
    assert "lora_a" not in injected["params"]["encoder"]["other"]

    # Perturb lora_b to test non-trivial delta merge
    injected["params"]["encoder"]["q_proj"]["lora_b"] = jnp.ones((4, 16))
    merged = merge_lora(injected)
    assert "lora_a" not in merged["params"]["encoder"]["q_proj"]
    assert not np.allclose(
        np.asarray(merged["params"]["encoder"]["q_proj"]["kernel"]),
        np.asarray(params["params"]["encoder"]["q_proj"]["kernel"]),
    )


def test_replace_head():
    params = {
        "params": {
            "backbone": {"weight": jnp.ones((10, 10))},
            "classifier": {"head": {"kernel": jnp.ones((10, 1000)), "bias": jnp.zeros((1000,))}},
        }
    }

    new_head_params = {"kernel": jnp.ones((10, 5)), "bias": jnp.zeros((5,))}
    updated = replace_head(params, "classifier/head", new_head_params)

    assert updated["params"]["classifier"]["head"]["kernel"].shape == (10, 5)
    assert updated["params"]["classifier"]["head"]["bias"].shape == (5,)
    assert updated["params"]["backbone"]["weight"].shape == (10, 10)


def test_trainable_param_mask():
    params = {
        "params": {
            "backbone": {
                "layer1": {"kernel": jnp.ones((4, 4))},
                "layer2": {"kernel": jnp.ones((4, 4))},
            },
            "head": {"kernel": jnp.ones((4, 2))},
        }
    }

    mask = get_trainable_param_mask(params, frozen_patterns=["backbone"], trainable_patterns=["head"])
    assert mask["params"]["backbone"]["layer1"]["kernel"] is False
    assert mask["params"]["backbone"]["layer2"]["kernel"] is False
    assert mask["params"]["head"]["kernel"] is True
