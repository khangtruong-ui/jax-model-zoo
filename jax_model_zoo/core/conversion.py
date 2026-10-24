"""Tensor layout and PyTree conversion utilities between PyTorch and JAX."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def torch_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a PyTorch tensor or array-like to a NumPy array on CPU."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.asarray(tensor)


def numpy_to_torch(arr: Any) -> Any:
    """Convert a NumPy array to a PyTorch tensor."""
    import torch
    if isinstance(arr, torch.Tensor):
        return arr
    if hasattr(arr, "device"):  # JAX array
        arr = np.asarray(arr)
    return torch.from_numpy(arr)


def conv2d_weight_torch_to_jax(w: np.ndarray) -> np.ndarray:
    """
    Convert PyTorch Conv2D weight to Flax Conv weight.
    PyTorch shape: [out_channels, in_channels, H, W]
    Flax shape:    [H, W, in_channels, out_channels]
    """
    w = torch_to_numpy(w)
    if w.ndim != 4:
        raise ValueError(f"Expected 4D Conv2d weight, got shape {w.shape}")
    return np.transpose(w, (2, 3, 1, 0))


def conv2d_weight_jax_to_torch(w: np.ndarray) -> np.ndarray:
    """
    Convert Flax Conv weight to PyTorch Conv2D weight.
    Flax shape:    [H, W, in_channels, out_channels]
    PyTorch shape: [out_channels, in_channels, H, W]
    """
    w = torch_to_numpy(w)
    if w.ndim != 4:
        raise ValueError(f"Expected 4D Flax Conv weight, got shape {w.shape}")
    return np.transpose(w, (3, 2, 0, 1))


def conv1d_weight_torch_to_jax(w: np.ndarray) -> np.ndarray:
    """
    Convert PyTorch Conv1D weight to Flax Conv weight.
    PyTorch shape: [out_channels, in_channels, L]
    Flax shape:    [L, in_channels, out_channels]
    """
    w = torch_to_numpy(w)
    if w.ndim != 3:
        raise ValueError(f"Expected 3D Conv1d weight, got shape {w.shape}")
    return np.transpose(w, (2, 1, 0))


def conv1d_weight_jax_to_torch(w: np.ndarray) -> np.ndarray:
    """
    Convert Flax Conv 1D weight to PyTorch Conv1D weight.
    Flax shape:    [L, in_channels, out_channels]
    PyTorch shape: [out_channels, in_channels, L]
    """
    w = torch_to_numpy(w)
    if w.ndim != 3:
        raise ValueError(f"Expected 3D Flax Conv1D weight, got shape {w.shape}")
    return np.transpose(w, (2, 1, 0))


def dense_weight_torch_to_jax(w: np.ndarray) -> np.ndarray:
    """
    Convert PyTorch Linear weight to Flax Dense kernel.
    PyTorch shape: [out_features, in_features]
    Flax shape:    [in_features, out_features]
    """
    w = torch_to_numpy(w)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {w.shape}")
    return np.transpose(w, (1, 0))


def dense_weight_jax_to_torch(w: np.ndarray) -> np.ndarray:
    """
    Convert Flax Dense kernel to PyTorch Linear weight.
    Flax shape:    [in_features, out_features]
    PyTorch shape: [out_features, in_features]
    """
    w = torch_to_numpy(w)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D Flax Dense kernel, got shape {w.shape}")
    return np.transpose(w, (1, 0))


def flatten_nested_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
) -> Dict[str, Any]:
    """Flatten a nested dictionary into single level with separated keys."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dict(
    d: Dict[str, Any],
    sep: str = "/",
) -> Dict[str, Any]:
    """Unflatten a single level dictionary with separated keys back to nested dict."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        curr = result
        for part in parts[:-1]:
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = value
    return result
