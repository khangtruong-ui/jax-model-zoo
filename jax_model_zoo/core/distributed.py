"""Multi-core GPU/TPU single-host parallelism utilities for JAX/Flax."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import flax.jax_utils as jutils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
try:
    from jax.sharding import PositionalSharding
except ImportError:
    PositionalSharding = None
import numpy as np


def get_devices() -> Sequence[jax.Device]:
    """Get list of available local devices (GPU or TPU cores)."""
    return jax.local_devices()


def get_device_count() -> int:
    """Get number of available local devices."""
    return jax.local_device_count()


def replicate(pytree: Any, devices: Optional[Sequence[jax.Device]] = None) -> Any:
    """
    Replicate PyTree across all local devices.
    Ensures safe broadcasting for pmap or data parallelism.
    """
    if devices is None:
        devices = get_devices()
    return jutils.replicate(pytree, devices=devices)


def unreplicate(pytree: Any) -> Any:
    """
    Extract a single copy of a replicated PyTree from the first device.
    """
    return jutils.unreplicate(pytree)


def shard_batch(
    batch: Union[jax.Array, np.ndarray, Dict[str, Any]],
    devices: Optional[Sequence[jax.Device]] = None,
) -> Any:
    """
    Shard a batch across devices by leading dimension:
    [B, ...] -> [num_devices, B // num_devices, ...]
    """
    num_devices = len(devices) if devices is not None else get_device_count()

    def _shard_arr(x: Any) -> jax.Array:
        arr = jnp.asarray(x)
        b_sz = arr.shape[0]
        if b_sz % num_devices != 0:
            raise ValueError(
                f"Batch size {b_sz} must be evenly divisible by device count {num_devices}"
            )
        per_dev = b_sz // num_devices
        return arr.reshape((num_devices, per_dev, *arr.shape[1:]))

    return jax.tree_util.tree_map(_shard_arr, batch)


def unshard_batch(
    sharded_batch: Union[jax.Array, Dict[str, Any]],
) -> Any:
    """
    Recombine sharded batch back to original batch dimension:
    [num_devices, per_dev, ...] -> [B, ...]
    """
    def _unshard_arr(x: Any) -> jax.Array:
        arr = jnp.asarray(x)
        num_devices, per_dev = arr.shape[0], arr.shape[1]
        return arr.reshape((num_devices * per_dev, *arr.shape[2:]))

    return jax.tree_util.tree_map(_unshard_arr, sharded_batch)


def create_data_parallel_apply(
    apply_fn: Callable[..., Any],
    axis_name: str = "batch",
) -> Callable[..., Any]:
    """
    Wrap an apply function with jax.pmap for distributed data-parallel execution
    across multiple GPU or TPU cores.
    """
    return jax.pmap(apply_fn, axis_name=axis_name)
