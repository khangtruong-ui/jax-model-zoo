"""Model surgery utilities: LoRA injection, head replacement, parameter freezing, and feature extraction."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.conversion import flatten_nested_dict, unflatten_nested_dict


class LoRADense(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Dense layer in Flax Linen.
    W = W_base + (alpha / rank) * (A @ B)
    """

    features: int
    rank: int = 4
    alpha: float = 1.0
    use_bias: bool = True
    trainable_base: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_features = x.shape[-1]
        scaling = self.alpha / float(self.rank)

        # Base weights
        w_base = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_features, self.features),
        )

        out = jnp.dot(x, w_base)

        if self.rank > 0:
            # LoRA A matrix (Gaussian initialized)
            lora_a = self.param(
                "lora_a",
                nn.initializers.normal(stddev=1.0 / np.sqrt(self.rank)),
                (in_features, self.rank),
            )
            # LoRA B matrix (Zero initialized)
            lora_b = self.param(
                "lora_b",
                nn.initializers.zeros,
                (self.rank, self.features),
            )
            lora_out = jnp.dot(jnp.dot(x, lora_a), lora_b) * scaling
            out = out + lora_out

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,))
            out = out + bias

        return out


class LoRAConv(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for 2D Convolutions in Flax Linen.
    Uses 1x1 depth-bottleneck factorization for efficient adaptation.
    """

    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]] = "SAME"
    rank: int = 4
    alpha: float = 1.0
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_features = x.shape[-1]
        scaling = self.alpha / float(self.rank)

        # Base conv
        base_conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            name="base_conv",
        )
        out = base_conv(x)

        if self.rank > 0:
            # Low-rank 1x1 adapter down-projection
            lora_a = self.param(
                "lora_a",
                nn.initializers.normal(stddev=1.0 / np.sqrt(self.rank)),
                (1, 1, in_features, self.rank),
            )
            # Spatial convolution or 1x1 up-projection
            lora_b = self.param(
                "lora_b",
                nn.initializers.zeros,
                (*self.kernel_size, self.rank, self.features),
            )

            # Down-project
            a_out = jax.lax.conv_general_dilated(
                x,
                lora_a,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            # Up-project with kernel strides
            b_out = jax.lax.conv_general_dilated(
                a_out,
                lora_b,
                window_strides=self.strides,
                padding=self.padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            out = out + b_out * scaling

        return out


def inject_lora(
    params: Dict[str, Any],
    target_patterns: Union[str, List[str]] = ("q_proj", "v_proj", "out_proj", "Dense"),
    rank: int = 4,
    alpha: float = 1.0,
    rng: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """
    Inject LoRA parameters into a parameter PyTree matching target patterns.
    """
    if isinstance(target_patterns, str):
        target_patterns = [target_patterns]

    if rng is None:
        rng = jax.random.PRNGKey(42)

    flat = flatten_nested_dict(params, sep="/")
    new_flat = dict(flat)

    for k, v in flat.items():
        if not k.endswith("/kernel"):
            continue

        parent_key = k[:-7]
        # Check if parent_key matches any of target patterns
        matches = any(re.search(pat, parent_key) for pat in target_patterns)
        if matches and v.ndim == 2:
            in_dim, out_dim = v.shape
            rng, rng_a = jax.random.split(rng)
            lora_a = jax.random.normal(rng_a, (in_dim, rank)) * (1.0 / np.sqrt(rank))
            lora_b = jnp.zeros((rank, out_dim), dtype=v.dtype)
            new_flat[f"{parent_key}/lora_a"] = lora_a
            new_flat[f"{parent_key}/lora_b"] = lora_b
            new_flat[f"{parent_key}/lora_scaling"] = jnp.array(alpha / float(rank), dtype=jnp.float32)

    return unflatten_nested_dict(new_flat, sep="/")


def merge_lora(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge injected LoRA weights (lora_a @ lora_b * scaling) back into base kernels
    and remove the lora keys.
    """
    flat = flatten_nested_dict(params, sep="/")
    new_flat = {}
    lora_parents = set()

    # Identify all modules that have LoRA weights
    for k in flat:
        if k.endswith("/lora_a"):
            lora_parents.add(k[:-7])

    for k, v in flat.items():
        if any(k.startswith(p + "/lora_") for p in lora_parents):
            continue  # strip lora keys

        parent = k[: k.rfind("/")] if "/" in k else ""
        if parent in lora_parents and k.endswith("/kernel"):
            lora_a = flat[f"{parent}/lora_a"]
            lora_b = flat[f"{parent}/lora_b"]
            scaling = flat.get(f"{parent}/lora_scaling", 1.0)
            delta = jnp.dot(lora_a, lora_b) * scaling
            new_flat[k] = v + delta
        else:
            new_flat[k] = v

    return unflatten_nested_dict(new_flat, sep="/")


def replace_head(
    params: Dict[str, Any],
    head_path: str,
    new_head_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Replace classification or task head in a nested parameter dictionary.
    head_path can be dot or slash separated (e.g. 'head' or 'classifier').
    """
    flat = flatten_nested_dict(params, sep="/")
    normalized_path = head_path.replace(".", "/").strip("/")

    # If normalized_path is not directly a prefix in flat, check if prefix exists with "params/"
    if not any(k == normalized_path or k.startswith(normalized_path + "/") for k in flat):
        if any(k.startswith(f"params/{normalized_path}/") or k == f"params/{normalized_path}" for k in flat):
            normalized_path = f"params/{normalized_path}"
        else:
            # Check if any path in flat matches normalized_path suffix / component
            matches = set()
            for k in flat:
                parts = k.split("/")
                target_parts = normalized_path.split("/")
                for i in range(len(parts) - len(target_parts) + 1):
                    if parts[i : i + len(target_parts)] == target_parts:
                        matches.add("/".join(parts[: i + len(target_parts)]))
            if len(matches) == 1:
                normalized_path = list(matches)[0]

    # Filter out existing head params
    filtered_flat = {
        k: v for k, v in flat.items()
        if not (k == normalized_path or k.startswith(normalized_path + "/"))
    }

    # Add new head params
    new_head_flat = flatten_nested_dict(new_head_params, sep="/")
    for k, v in new_head_flat.items():
        full_key = f"{normalized_path}/{k}" if k else normalized_path
        filtered_flat[full_key] = v

    return unflatten_nested_dict(filtered_flat, sep="/")


def get_trainable_param_mask(
    params: Dict[str, Any],
    trainable_patterns: Optional[Sequence[str]] = None,
    frozen_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Create a boolean mask PyTree for Optax (e.g. `optax.masked`) where True
    indicates trainable parameters and False indicates frozen parameters.
    """
    flat = flatten_nested_dict(params, sep="/")
    mask_flat: Dict[str, bool] = {}

    for k in flat:
        # Default: trainable
        is_trainable = True

        if frozen_patterns:
            if any(re.search(pat, k) for pat in frozen_patterns):
                is_trainable = False

        if trainable_patterns:
            if any(re.search(pat, k) for pat in trainable_patterns):
                is_trainable = True

        mask_flat[k] = is_trainable

    return unflatten_nested_dict(mask_flat, sep="/")
