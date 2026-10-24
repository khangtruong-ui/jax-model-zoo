"""Tests for FLOPs exporting and compute cost analysis."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from jax_model_zoo.core.flops import count_parameters, export_flops


class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        h = jax.nn.relu(h)
        h = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(h)
        h = jax.nn.relu(h)
        h = jnp.mean(h, axis=(1, 2))
        return nn.Dense(features=10)(h)


def test_export_flops_cnn():
    model = SimpleCNN()
    x = jnp.ones((1, 32, 32, 3))
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, x)

    report = export_flops(lambda p, v: model.apply(p, v), params, x, verbose=False)

    assert report.total_flops > 0
    assert report.total_macs > 0
    assert report.gflops >= 0.0
    assert report.param_count > 0
    assert "conv_general_dilated" in report.op_breakdown
    assert "dot_general" in report.op_breakdown

    d = report.to_dict()
    assert "total_flops" in d
    assert "gflops" in d

    summary_str = report.summary()
    assert "FLOPs & Compute Cost Report" in summary_str


def test_count_parameters():
    params = {
        "params": {
            "w1": jnp.ones((10, 20)),
            "b1": jnp.zeros((20,)),
            "w2": jnp.ones((20, 5)),
        }
    }
    # 200 + 20 + 100 = 320
    assert count_parameters(params) == 320
