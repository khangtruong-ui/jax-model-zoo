"""Device-topology agnostic checkpoint saving and loading for JAX/Flax models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file as load_safetensors_flax
from safetensors.flax import save_file as save_safetensors_flax
from safetensors.numpy import load_file as load_safetensors_np
from safetensors.numpy import save_file as save_safetensors_np

from jax_model_zoo.core.conversion import flatten_nested_dict, unflatten_nested_dict

logger = logging.getLogger("jax_model_zoo.core.checkpoint")


def _sanitize_tree_to_numpy(pytree: Any) -> Any:
    """Recursively convert all leaves to host NumPy arrays to strip device sharding."""
    def _to_np(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        # Ensure it is brought back to CPU and stripped of sharding
        return np.asarray(jax.device_get(x))

    return jax.tree_util.tree_map(_to_np, pytree)


def _convert_numpy_tree_to_jax(pytree: Any, device: Optional[Any] = None) -> Any:
    """Convert numpy array leaves to jax.Array on target device (or default)."""
    def _to_jax(x: Any) -> jax.Array:
        if isinstance(x, (np.ndarray, list, tuple, int, float)):
            arr = jnp.asarray(x)
            if device is not None:
                arr = jax.device_put(arr, device)
            return arr
        return x

    return jax.tree_util.tree_map(_to_jax, pytree)


def save_safetensors(
    params: Dict[str, Any],
    path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Save nested JAX parameters to a .safetensors file.
    Safetensors format is inherently topology-agnostic, zero-copy, and fast.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean device sharding by converting to host numpy
    sanitized = _sanitize_tree_to_numpy(params)
    flat_dict = flatten_nested_dict(sanitized, sep=".")
    
    # Ensure all values are contiguous numpy arrays
    tensors = {k: np.ascontiguousarray(v) for k, v in flat_dict.items()}

    meta = metadata.copy() if metadata else {}
    meta["format"] = "jax_model_zoo_safetensors"

    save_safetensors_np(tensors, str(path), metadata=meta)
    return path


def load_safetensors(
    path: Union[str, Path],
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load nested JAX parameters from a .safetensors file onto target device.
    Topology-agnostic: loads correctly whether saved on 1 GPU, 8 GPUs, or TPU.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    tensors = load_safetensors_np(str(path))
    nested = unflatten_nested_dict(tensors, sep=".")
    return _convert_numpy_tree_to_jax(nested, device=device)


def save_checkpoint(
    params: Dict[str, Any],
    path: Union[str, Path],
    format: str = "safetensors",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save parameters in a device-topology agnostic format.
    Supports 'safetensors', 'msgpack' (Flax), and 'orbax'.
    """
    path = Path(path)
    format = format.lower()

    meta_str = {k: str(v) for k, v in (metadata or {}).items()}

    if format == "safetensors":
        if not path.suffix:
            path = path.with_suffix(".safetensors")
        return save_safetensors(params, path, metadata=meta_str)

    elif format == "msgpack":
        if not path.suffix:
            path = path.with_suffix(".msgpack")
        path.parent.mkdir(parents=True, exist_ok=True)
        from flax.serialization import to_bytes
        sanitized = _sanitize_tree_to_numpy(params)
        bytes_data = to_bytes(sanitized)
        with open(path, "wb") as f:
            f.write(bytes_data)
        if metadata:
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        return path

    elif format == "orbax":
        # Save orbax checkpoint in a topology-agnostic way (pure CPU leaves)
        import orbax.checkpoint as ocp
        sanitized = _sanitize_tree_to_numpy(params)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path.resolve(), sanitized)
        checkpointer.wait_until_finished()
        return path

    else:
        raise ValueError(f"Unsupported checkpoint format: {format}")


def load_checkpoint(
    path: Union[str, Path],
    format: str = "auto",
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint in a device-topology agnostic manner.
    Automatically detects format if format='auto'.
    """
    path = Path(path)
    if not path.exists():
        # Try appending .safetensors or .msgpack
        if path.with_suffix(".safetensors").exists():
            path = path.with_suffix(".safetensors")
        elif path.with_suffix(".msgpack").exists():
            path = path.with_suffix(".msgpack")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {path}")

    if format == "auto":
        if path.suffix == ".safetensors":
            format = "safetensors"
        elif path.suffix == ".msgpack":
            format = "msgpack"
        elif path.is_dir():
            format = "orbax"
        else:
            format = "safetensors"

    if format == "safetensors":
        return load_safetensors(path, device=device)

    elif format == "msgpack":
        from flax.serialization import from_bytes
        with open(path, "rb") as f:
            bytes_data = f.read()
        sanitized = from_bytes(None, bytes_data)
        return _convert_numpy_tree_to_jax(sanitized, device=device)

    elif format == "orbax":
        import orbax.checkpoint as ocp
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(path.resolve())
        return _convert_numpy_tree_to_jax(restored, device=device)

    else:
        raise ValueError(f"Unknown checkpoint format: {format}")


class DeviceAgnosticCheckpointer:
    """
    High-level checkpoint manager for saving and restoring model parameters
    across varying device topologies (1 GPU, 8 GPUs, TPU, CPU).
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        default_format: str = "safetensors",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.default_format = default_format

    def save(
        self,
        step_or_name: Union[int, str],
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
    ) -> Path:
        """Save a named or stepped checkpoint."""
        fmt = format or self.default_format
        prefix = f"checkpoint_{step_or_name}" if isinstance(step_or_name, int) else str(step_or_name)
        ext = ".safetensors" if fmt == "safetensors" else (".msgpack" if fmt == "msgpack" else "")
        target_path = self.checkpoint_dir / f"{prefix}{ext}"
        return save_checkpoint(params, target_path, format=fmt, metadata=metadata)

    def load(
        self,
        step_or_name: Union[int, str],
        format: str = "auto",
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load a named or stepped checkpoint."""
        prefix = f"checkpoint_{step_or_name}" if isinstance(step_or_name, int) else str(step_or_name)
        # Search for exact file or pattern
        candidates = list(self.checkpoint_dir.glob(f"{prefix}*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found matching {prefix} in {self.checkpoint_dir}")
        target_path = candidates[0]
        return load_checkpoint(target_path, format=format, device=device)
