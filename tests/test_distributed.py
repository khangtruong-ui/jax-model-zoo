"""Tests for multi-core GPU/TPU distribution helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_model_zoo.core.distributed import (
    get_device_count,
    get_devices,
    replicate,
    shard_batch,
    unreplicate,
    unshard_batch,
)


def test_device_discovery():
    devices = get_devices()
    count = get_device_count()
    assert len(devices) == count
    assert count >= 1


def test_batch_sharding_roundtrip():
    # Test batch size 4 on available devices (if 1 device, shards to 1x4)
    num_dev = get_device_count()
    b_sz = num_dev * 2
    x = jnp.arange(b_sz * 8).reshape((b_sz, 8))

    sharded = shard_batch(x)
    assert sharded.shape == (num_dev, 2, 8)

    unsharded = unshard_batch(sharded)
    assert unsharded.shape == (b_sz, 8)
    np.testing.assert_array_equal(np.asarray(unsharded), np.asarray(x))


def test_replicate_unreplicate():
    pytree = {"w": jnp.ones((4, 4)), "b": jnp.zeros((4,))}
    replicated = replicate(pytree)
    unrep = unreplicate(replicated)
    np.testing.assert_array_equal(np.asarray(unrep["w"]), np.asarray(pytree["w"]))
