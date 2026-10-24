"""Tests for topology-agnostic checkpoint saving and loading."""

import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.checkpoint import (
    DeviceAgnosticCheckpointer,
    load_checkpoint,
    load_safetensors,
    save_checkpoint,
    save_safetensors,
)


@pytest.fixture
def sample_params():
    rng = jax.random.PRNGKey(0)
    return {
        "params": {
            "encoder": {
                "kernel": jax.random.normal(rng, (16, 32)),
                "bias": jnp.zeros((32,)),
            },
            "decoder": {
                "weight": jax.random.normal(rng, (3, 3, 32, 64)),
                "norm": {
                    "scale": jnp.ones((64,)),
                    "bias": jnp.zeros((64,)),
                },
            },
        }
    }


def test_safetensors_save_and_load(sample_params):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.safetensors"
        save_safetensors(sample_params, path, metadata={"creator": "jax_model_zoo"})
        assert path.exists()

        restored = load_safetensors(path)
        assert "params" in restored
        assert "encoder" in restored["params"]
        assert "kernel" in restored["params"]["encoder"]
        np.testing.assert_allclose(
            np.asarray(restored["params"]["encoder"]["kernel"]),
            np.asarray(sample_params["params"]["encoder"]["kernel"]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_checkpoint_msgpack_and_orbax(sample_params):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Msgpack format
        msgpack_path = Path(tmpdir) / "model.msgpack"
        save_checkpoint(sample_params, msgpack_path, format="msgpack")
        assert msgpack_path.exists()

        restored_mp = load_checkpoint(msgpack_path, format="msgpack")
        np.testing.assert_allclose(
            np.asarray(restored_mp["params"]["encoder"]["kernel"]),
            np.asarray(sample_params["params"]["encoder"]["kernel"]),
            rtol=1e-5,
        )

        # Orbax format
        orbax_dir = Path(tmpdir) / "orbax_ckpt"
        save_checkpoint(sample_params, orbax_dir, format="orbax")
        assert orbax_dir.exists()

        restored_orbax = load_checkpoint(orbax_dir, format="orbax")
        np.testing.assert_allclose(
            np.asarray(restored_orbax["params"]["encoder"]["kernel"]),
            np.asarray(sample_params["params"]["encoder"]["kernel"]),
            rtol=1e-5,
        )


def test_device_agnostic_checkpointer(sample_params):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = DeviceAgnosticCheckpointer(checkpoint_dir=tmpdir, default_format="safetensors")

        saved_path = ckpt_mgr.save(100, sample_params, metadata={"loss": 0.42})
        assert saved_path.exists()

        loaded_params = ckpt_mgr.load(100)
        assert "params" in loaded_params
        np.testing.assert_allclose(
            np.asarray(loaded_params["params"]["decoder"]["weight"]),
            np.asarray(sample_params["params"]["decoder"]["weight"]),
            rtol=1e-5,
        )


def test_topology_invariance_cpu_load(sample_params):
    """Simulate loading parameters saved on one device topology onto CPU device."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "topo_test.safetensors"
        save_safetensors(sample_params, path)

        cpu_device = jax.devices("cpu")[0]
        restored = load_safetensors(path, device=cpu_device)

        leaf = jax.tree_util.tree_leaves(restored)[0]
        assert leaf.device == cpu_device
