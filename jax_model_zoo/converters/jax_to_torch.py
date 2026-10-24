"""Converters from Flax Linen parameter PyTree to PyTorch state_dict."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np

from jax_model_zoo.core.conversion import (
    conv2d_weight_jax_to_torch,
    dense_weight_jax_to_torch,
    flatten_nested_dict,
    numpy_to_torch,
    torch_to_numpy,
)


def convert_jax_vit_to_torch(
    jax_params: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert Flax ViT parameters to PyTorch ViT state dict."""
    raw = jax_params.get("params", jax_params)
    flat = flatten_nested_dict(raw, sep="/")
    torch_state: Dict[str, Any] = {}

    for k, v in flat.items():
        v_np = torch_to_numpy(v)

        if k == "patch_embed/projection/kernel":
            torch_state["vit.embeddings.patch_embeddings.projection.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
        elif k == "patch_embed/projection/bias":
            torch_state["vit.embeddings.patch_embeddings.projection.bias"] = numpy_to_torch(v_np)
        elif k == "cls_token":
            torch_state["vit.embeddings.cls_token"] = numpy_to_torch(v_np.reshape((1, 1, -1)))
        elif k == "pos_embed":
            torch_state["vit.embeddings.position_embeddings"] = numpy_to_torch(v_np.reshape((1, -1, v_np.shape[-1])))
        elif k == "norm/scale":
            torch_state["vit.layernorm.weight"] = numpy_to_torch(v_np)
        elif k == "norm/bias":
            torch_state["vit.layernorm.bias"] = numpy_to_torch(v_np)
        elif k == "head/kernel":
            torch_state["classifier.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
        elif k == "head/bias":
            torch_state["classifier.bias"] = numpy_to_torch(v_np)

        elif k.startswith("block_"):
            parts = k.split("/")
            block_idx = parts[0].split("_")[1]
            sub = "/".join(parts[1:])

            if sub == "norm1/scale":
                torch_state[f"vit.encoder.layer.{block_idx}.layernorm_before.weight"] = numpy_to_torch(v_np)
            elif sub == "norm1/bias":
                torch_state[f"vit.encoder.layer.{block_idx}.layernorm_before.bias"] = numpy_to_torch(v_np)
            elif sub == "attn/out_proj/kernel":
                torch_state[f"vit.encoder.layer.{block_idx}.attention.output.dense.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
            elif sub == "attn/out_proj/bias":
                torch_state[f"vit.encoder.layer.{block_idx}.attention.output.dense.bias"] = numpy_to_torch(v_np)
            elif sub == "norm2/scale":
                torch_state[f"vit.encoder.layer.{block_idx}.layernorm_after.weight"] = numpy_to_torch(v_np)
            elif sub == "norm2/bias":
                torch_state[f"vit.encoder.layer.{block_idx}.layernorm_after.bias"] = numpy_to_torch(v_np)
            elif sub == "mlp/fc1/kernel":
                torch_state[f"vit.encoder.layer.{block_idx}.intermediate.dense.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
            elif sub == "mlp/fc1/bias":
                torch_state[f"vit.encoder.layer.{block_idx}.intermediate.dense.bias"] = numpy_to_torch(v_np)
            elif sub == "mlp/fc2/kernel":
                torch_state[f"vit.encoder.layer.{block_idx}.output.dense.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
            elif sub == "mlp/fc2/bias":
                torch_state[f"vit.encoder.layer.{block_idx}.output.dense.bias"] = numpy_to_torch(v_np)
            elif sub == "attn/qkv/kernel":
                # Split qkv kernel [dim, 3 * dim] -> q, k, v each [dim, dim]
                qkv_t = dense_weight_jax_to_torch(v_np)
                q, k, v = np.split(qkv_t, 3, axis=0)
                torch_state[f"vit.encoder.layer.{block_idx}.attention.attention.query.weight"] = numpy_to_torch(q)
                torch_state[f"vit.encoder.layer.{block_idx}.attention.attention.key.weight"] = numpy_to_torch(k)
                torch_state[f"vit.encoder.layer.{block_idx}.attention.attention.value.weight"] = numpy_to_torch(v)
            elif sub == "attn/qkv/bias":
                q_b, k_b, v_b = np.split(v_np, 3, axis=0)
                torch_state[f"vit.encoder.layer.{block_idx}.attention.attention.query.bias"] = numpy_to_torch(q_b)
                torch_state[f"vit.encoder.layer.{block_idx}.attention.attention.key.bias"] = numpy_to_torch(k_b)
                torch_state[f"vit.encoder.layer.{block_idx}.attention.attention.value.bias"] = numpy_to_torch(v_b)

    return torch_state


def convert_jax_diffusion_diff_to_torch(
    jax_params: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert Flax DiffusionDiff parameters to PyTorch state dict matching ai-detection-unet-gpu."""
    raw = jax_params.get("params", jax_params)
    flat = flatten_nested_dict(raw, sep="/")
    torch_state: Dict[str, Any] = {}

    for k, v in flat.items():
        v_np = torch_to_numpy(v)

        # Z normalization
        if k == "z_norm/scale":
            torch_state["z_norm.weight"] = numpy_to_torch(v_np)
        elif k == "z_norm/bias":
            torch_state["z_norm.bias"] = numpy_to_torch(v_np)

        # Init conv
        elif k == "decoder/init_conv_0/kernel":
            torch_state["decoder.init_conv.0.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
        elif k == "decoder/init_conv_1/scale":
            torch_state["decoder.init_conv.1.weight"] = numpy_to_torch(v_np)
        elif k == "decoder/init_conv_1/bias":
            torch_state["decoder.init_conv.1.bias"] = numpy_to_torch(v_np)

        # Output conv
        elif k == "decoder/out_conv/kernel":
            torch_state["decoder.out_conv.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
        elif k == "decoder/out_conv/bias":
            torch_state["decoder.out_conv.bias"] = numpy_to_torch(v_np)

        # Classifier head
        elif k == "classifier_head/fc1/kernel":
            torch_state["classifier_head.fc1.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
        elif k == "classifier_head/fc1/bias":
            torch_state["classifier_head.fc1.bias"] = numpy_to_torch(v_np)
        elif k == "classifier_head/fc2/kernel":
            torch_state["classifier_head.fc2.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
        elif k == "classifier_head/fc2/bias":
            torch_state["classifier_head.fc2.bias"] = numpy_to_torch(v_np)

        # Stages
        elif k.startswith("decoder/stages_"):
            parts = k.split("/")
            stage_idx = parts[1].split("_")[1]
            sub = "/".join(parts[2:])

            if sub == "up_conv/kernel":
                torch_state[f"decoder.stages.{stage_idx}.up.1.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
            elif sub == "norm/scale":
                torch_state[f"decoder.stages.{stage_idx}.norm.weight"] = numpy_to_torch(v_np)
            elif sub == "norm/bias":
                torch_state[f"decoder.stages.{stage_idx}.norm.bias"] = numpy_to_torch(v_np)
            elif "res_blocks_" in sub:
                res_parts = sub.split("/")
                res_idx = res_parts[0].split("_")[2]
                res_leaf = res_parts[1]
                if res_leaf == "conv1/kernel":
                    torch_state[f"decoder.stages.{stage_idx}.res_blocks.{res_idx}.conv1.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
                elif res_leaf == "norm1/scale":
                    torch_state[f"decoder.stages.{stage_idx}.res_blocks.{res_idx}.norm1.weight"] = numpy_to_torch(v_np)
                elif res_leaf == "norm1/bias":
                    torch_state[f"decoder.stages.{stage_idx}.res_blocks.{res_idx}.norm1.bias"] = numpy_to_torch(v_np)
                elif res_leaf == "conv2/kernel":
                    torch_state[f"decoder.stages.{stage_idx}.res_blocks.{res_idx}.conv2.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
                elif res_leaf == "norm2/scale":
                    torch_state[f"decoder.stages.{stage_idx}.res_blocks.{res_idx}.norm2.weight"] = numpy_to_torch(v_np)
                elif res_leaf == "norm2/bias":
                    torch_state[f"decoder.stages.{stage_idx}.res_blocks.{res_idx}.norm2.bias"] = numpy_to_torch(v_np)

        # Skip fusions
        elif k.startswith("decoder/skip_fusions_"):
            parts = k.split("/")
            f_idx = parts[1].split("_")[2]
            sub = parts[2]
            if sub == "conv/kernel":
                torch_state[f"decoder.skip_fusions.{f_idx}.conv.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
            elif sub == "norm/scale":
                torch_state[f"decoder.skip_fusions.{f_idx}.norm.weight"] = numpy_to_torch(v_np)
            elif sub == "norm/bias":
                torch_state[f"decoder.skip_fusions.{f_idx}.norm.bias"] = numpy_to_torch(v_np)
            elif sub == "skip_norm/scale":
                torch_state[f"decoder.skip_fusions.{f_idx}.skip_norm.weight"] = numpy_to_torch(v_np)
            elif sub == "skip_norm/bias":
                torch_state[f"decoder.skip_fusions.{f_idx}.skip_norm.bias"] = numpy_to_torch(v_np)

        # Perpendicular skip fusions
        elif k.startswith("decoder/perp_skip_fusions_"):
            parts = k.split("/")
            f_idx = parts[1].split("_")[3]
            sub = parts[2]
            if sub == "conv/kernel":
                torch_state[f"decoder.perp_skip_fusions.{f_idx}.conv.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
            elif sub == "norm/scale":
                torch_state[f"decoder.perp_skip_fusions.{f_idx}.norm.weight"] = numpy_to_torch(v_np)
            elif sub == "norm/bias":
                torch_state[f"decoder.perp_skip_fusions.{f_idx}.norm.bias"] = numpy_to_torch(v_np)
            elif sub == "perp_norm/scale":
                torch_state[f"decoder.perp_skip_fusions.{f_idx}.perp_norm.weight"] = numpy_to_torch(v_np)
            elif sub == "perp_norm/bias":
                torch_state[f"decoder.perp_skip_fusions.{f_idx}.perp_norm.bias"] = numpy_to_torch(v_np)

    return torch_state


def convert_jax_params_to_torch(
    jax_params: Dict[str, Any],
    model_type: str,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Universal dispatcher for converting Flax Linen parameters to PyTorch state dict."""
    model_type = model_type.lower()
    if model_type in ("vit", "vit_b", "vit_l", "vision_transformer"):
        return convert_jax_vit_to_torch(jax_params, config)
    elif model_type in ("diffusion_diff", "diffusion_diff_v2", "diffusion_diff_minimized"):
        return convert_jax_diffusion_diff_to_torch(jax_params, config)
    else:
        # Default recursive tensor transposition
        raw = jax_params.get("params", jax_params)
        flat = flatten_nested_dict(raw, sep=".")
        torch_state: Dict[str, Any] = {}
        for k, v in flat.items():
            v_np = torch_to_numpy(v)
            if k.endswith(".kernel"):
                base_k = k[:-7]
                if v_np.ndim == 4:
                    torch_state[f"{base_k}.weight"] = numpy_to_torch(conv2d_weight_jax_to_torch(v_np))
                elif v_np.ndim == 2:
                    torch_state[f"{base_k}.weight"] = numpy_to_torch(dense_weight_jax_to_torch(v_np))
                else:
                    torch_state[f"{base_k}.weight"] = numpy_to_torch(v_np)
            elif k.endswith(".scale"):
                torch_state[f"{k[:-6]}.weight"] = numpy_to_torch(v_np)
            else:
                torch_state[k] = numpy_to_torch(v_np)
        return torch_state
