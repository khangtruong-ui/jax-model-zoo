"""Converters from PyTorch checkpoint state_dict to Flax Linen parameter PyTree."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import math
import numpy as np

from jax_model_zoo.core.conversion import (
    conv2d_weight_torch_to_jax,
    dense_weight_torch_to_jax,
    flatten_nested_dict,
    torch_to_numpy,
    unflatten_nested_dict,
)


def convert_torch_vit_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert HuggingFace/timm ViT PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        # 1. Patch embeddings
        if "patch_embeddings.projection.weight" in k or "patch_embed.proj.weight" in k or "patch_embed.projection.weight" in k:
            jax_params["patch_embed/projection/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "patch_embeddings.projection.bias" in k or "patch_embed.proj.bias" in k or "patch_embed.projection.bias" in k:
            jax_params["patch_embed/projection/bias"] = v_np

        # 2. CLS and Positional embeddings
        elif k.endswith("cls_token") or "embeddings.cls_token" in k:
            jax_params["cls_token"] = v_np.reshape((1, 1, -1))
        elif k.endswith("position_embeddings") or k.endswith("pos_embed"):
            jax_params["pos_embed"] = v_np.reshape((1, -1, v_np.shape[-1]))

        # 3. Final LayerNorm
        elif k in ("norm.weight", "vit.layernorm.weight") or (k.endswith("layernorm.weight") and "layers" not in k and "layer" not in k):
            jax_params["norm/scale"] = v_np
        elif k in ("norm.bias", "vit.layernorm.bias") or (k.endswith("layernorm.bias") and "layers" not in k and "layer" not in k):
            jax_params["norm/bias"] = v_np

        # 4. Classifier Head
        elif "classifier.weight" in k or "head.weight" in k:
            jax_params["head/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "classifier.bias" in k or "head.bias" in k:
            jax_params["head/bias"] = v_np

    # 5. Transformer Blocks
    depth = (config.depth if config else 24)
    for b_idx in range(depth):
        prefix = f"block_{b_idx}"

        # LayerNorm 1
        for k in [
            f"vit.layers.{b_idx}.layernorm_before.weight",
            f"vit.encoder.layer.{b_idx}.layernorm_before.weight",
            f"layer.{b_idx}.layernorm_before.weight",
            f"blocks.{b_idx}.norm1.weight",
        ]:
            if k in torch_state:
                jax_params[f"{prefix}/norm1/scale"] = torch_to_numpy(torch_state[k])
                break
        for k in [
            f"vit.layers.{b_idx}.layernorm_before.bias",
            f"vit.encoder.layer.{b_idx}.layernorm_before.bias",
            f"layer.{b_idx}.layernorm_before.bias",
            f"blocks.{b_idx}.norm1.bias",
        ]:
            if k in torch_state:
                jax_params[f"{prefix}/norm1/bias"] = torch_to_numpy(torch_state[k])
                break

        # LayerNorm 2
        for k in [
            f"vit.layers.{b_idx}.layernorm_after.weight",
            f"vit.encoder.layer.{b_idx}.layernorm_after.weight",
            f"layer.{b_idx}.layernorm_after.weight",
            f"blocks.{b_idx}.norm2.weight",
        ]:
            if k in torch_state:
                jax_params[f"{prefix}/norm2/scale"] = torch_to_numpy(torch_state[k])
                break
        for k in [
            f"vit.layers.{b_idx}.layernorm_after.bias",
            f"vit.encoder.layer.{b_idx}.layernorm_after.bias",
            f"layer.{b_idx}.layernorm_after.bias",
            f"blocks.{b_idx}.norm2.bias",
        ]:
            if k in torch_state:
                jax_params[f"{prefix}/norm2/bias"] = torch_to_numpy(torch_state[k])
                break

        # Attention QKV
        q_k = [k for k in torch_state if (
            f"layers.{b_idx}.attention.q_proj.weight" in k or
            f"layer.{b_idx}.attention.attention.query.weight" in k or
            f"layer.{b_idx}.attention.q_proj.weight" in k
        )]
        k_k = [k for k in torch_state if (
            f"layers.{b_idx}.attention.k_proj.weight" in k or
            f"layer.{b_idx}.attention.attention.key.weight" in k or
            f"layer.{b_idx}.attention.k_proj.weight" in k
        )]
        v_k = [k for k in torch_state if (
            f"layers.{b_idx}.attention.v_proj.weight" in k or
            f"layer.{b_idx}.attention.attention.value.weight" in k or
            f"layer.{b_idx}.attention.v_proj.weight" in k
        )]

        if q_k and k_k and v_k:
            q_w = torch_to_numpy(torch_state[q_k[0]])
            k_w = torch_to_numpy(torch_state[k_k[0]])
            v_w = torch_to_numpy(torch_state[v_k[0]])
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(qkv_w)

            q_b = [k for k in torch_state if (
                f"layers.{b_idx}.attention.q_proj.bias" in k or
                f"layer.{b_idx}.attention.attention.query.bias" in k or
                f"layer.{b_idx}.attention.q_proj.bias" in k
            )]
            k_b = [k for k in torch_state if (
                f"layers.{b_idx}.attention.k_proj.bias" in k or
                f"layer.{b_idx}.attention.attention.key.bias" in k or
                f"layer.{b_idx}.attention.k_proj.bias" in k
            )]
            v_b = [k for k in torch_state if (
                f"layers.{b_idx}.attention.v_proj.bias" in k or
                f"layer.{b_idx}.attention.attention.value.bias" in k or
                f"layer.{b_idx}.attention.v_proj.bias" in k
            )]
            if q_b and k_b and v_b:
                qkv_b = np.concatenate([
                    torch_to_numpy(torch_state[q_b[0]]),
                    torch_to_numpy(torch_state[k_b[0]]),
                    torch_to_numpy(torch_state[v_b[0]]),
                ], axis=0)
                jax_params[f"{prefix}/attn/qkv/bias"] = qkv_b
        elif any(f"blocks.{b_idx}.attn.qkv.weight" in k for k in torch_state):
            w = torch_to_numpy([torch_state[k] for k in torch_state if f"blocks.{b_idx}.attn.qkv.weight" in k][0])
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(w)
            b_list = [torch_state[k] for k in torch_state if f"blocks.{b_idx}.attn.qkv.bias" in k]
            if b_list:
                jax_params[f"{prefix}/attn/qkv/bias"] = torch_to_numpy(b_list[0])

        # Attention Out Proj
        out_k = [k for k in torch_state if (
            f"layers.{b_idx}.attention.o_proj.weight" in k or
            f"layer.{b_idx}.attention.output.dense.weight" in k or
            f"layer.{b_idx}.attention.o_proj.weight" in k or
            f"blocks.{b_idx}.attn.proj.weight" in k
        )]
        if out_k:
            jax_params[f"{prefix}/attn/out_proj/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[out_k[0]]))
        out_b = [k for k in torch_state if (
            f"layers.{b_idx}.attention.o_proj.bias" in k or
            f"layer.{b_idx}.attention.output.dense.bias" in k or
            f"layer.{b_idx}.attention.o_proj.bias" in k or
            f"blocks.{b_idx}.attn.proj.bias" in k
        )]
        if out_b:
            jax_params[f"{prefix}/attn/out_proj/bias"] = torch_to_numpy(torch_state[out_b[0]])

        # MLP
        fc1_w = [k for k in torch_state if (
            f"layers.{b_idx}.mlp.fc1.weight" in k or
            f"layer.{b_idx}.intermediate.dense.weight" in k or
            f"layer.{b_idx}.mlp.fc1.weight" in k or
            f"blocks.{b_idx}.mlp.fc1.weight" in k
        )]
        if fc1_w:
            jax_params[f"{prefix}/mlp/fc1/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[fc1_w[0]]))
        fc1_b = [k for k in torch_state if (
            f"layers.{b_idx}.mlp.fc1.bias" in k or
            f"layer.{b_idx}.intermediate.dense.bias" in k or
            f"layer.{b_idx}.mlp.fc1.bias" in k or
            f"blocks.{b_idx}.mlp.fc1.bias" in k
        )]
        if fc1_b:
            jax_params[f"{prefix}/mlp/fc1/bias"] = torch_to_numpy(torch_state[fc1_b[0]])

        fc2_w = [k for k in torch_state if (
            f"layers.{b_idx}.mlp.fc2.weight" in k or
            f"layer.{b_idx}.output.dense.weight" in k or
            f"layer.{b_idx}.mlp.fc2.weight" in k or
            f"blocks.{b_idx}.mlp.fc2.weight" in k
        )]
        if fc2_w:
            jax_params[f"{prefix}/mlp/fc2/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[fc2_w[0]]))
        fc2_b = [k for k in torch_state if (
            f"layers.{b_idx}.mlp.fc2.bias" in k or
            f"layer.{b_idx}.output.dense.bias" in k or
            f"layer.{b_idx}.mlp.fc2.bias" in k or
            f"blocks.{b_idx}.mlp.fc2.bias" in k
        )]
        if fc2_b:
            jax_params[f"{prefix}/mlp/fc2/bias"] = torch_to_numpy(torch_state[fc2_b[0]])

    nested = unflatten_nested_dict(jax_params, sep="/")
    return {"params": nested}


def convert_torch_mae_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert HuggingFace/official MAE PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        if "patch_embeddings.projection.weight" in k or "patch_embed.proj.weight" in k or "patch_embed.projection.weight" in k:
            jax_params["patch_embed/projection/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "patch_embeddings.projection.bias" in k or "patch_embed.proj.bias" in k or "patch_embed.projection.bias" in k:
            jax_params["patch_embed/projection/bias"] = v_np
        elif k.endswith("cls_token") or "embeddings.cls_token" in k:
            jax_params["cls_token"] = v_np.reshape((1, 1, -1))
        elif k.endswith("position_embeddings") or (k.endswith("pos_embed") and "decoder" not in k):
            jax_params["pos_embed"] = v_np.reshape((1, -1, v_np.shape[-1]))
        elif k.endswith("mask_token"):
            jax_params["mask_token"] = v_np.reshape((1, 1, -1))
        elif "decoder_pos_embed" in k:
            jax_params["decoder_pos_embed"] = v_np.reshape((1, -1, v_np.shape[-1]))
        elif "decoder_embed.weight" in k:
            jax_params["decoder_embed/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "decoder_embed.bias" in k:
            jax_params["decoder_embed/bias"] = v_np
        elif "decoder_pred.weight" in k:
            jax_params["decoder_pred/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "decoder_pred.bias" in k:
            jax_params["decoder_pred/bias"] = v_np
        elif (k in ("vit.layernorm.weight", "norm.weight")) or ("layernorm.weight" in k and "decoder" not in k and "layer" not in k):
            jax_params["encoder_norm/scale"] = v_np
        elif (k in ("vit.layernorm.bias", "norm.bias")) or ("layernorm.bias" in k and "decoder" not in k and "layer" not in k):
            jax_params["encoder_norm/bias"] = v_np
        elif "decoder_norm.weight" in k:
            jax_params["decoder_norm/scale"] = v_np
        elif "decoder_norm.bias" in k:
            jax_params["decoder_norm/bias"] = v_np

    # Encoder transformer blocks
    enc_depth = (config.depth if config else 12)
    for b_idx in range(enc_depth):
        prefix = f"encoder_block_{b_idx}"
        for k in [f"vit.layers.{b_idx}.layernorm_before.weight", f"blocks.{b_idx}.norm1.weight", f"layer.{b_idx}.layernorm_before.weight"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm1/scale"] = torch_to_numpy(torch_state[k])
                break
        for k in [f"vit.layers.{b_idx}.layernorm_before.bias", f"blocks.{b_idx}.norm1.bias", f"layer.{b_idx}.layernorm_before.bias"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm1/bias"] = torch_to_numpy(torch_state[k])
                break
        for k in [f"vit.layers.{b_idx}.layernorm_after.weight", f"blocks.{b_idx}.norm2.weight", f"layer.{b_idx}.layernorm_after.weight"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm2/scale"] = torch_to_numpy(torch_state[k])
                break
        for k in [f"vit.layers.{b_idx}.layernorm_after.bias", f"blocks.{b_idx}.norm2.bias", f"layer.{b_idx}.layernorm_after.bias"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm2/bias"] = torch_to_numpy(torch_state[k])
                break

        q_k = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.q_proj.weight" in k or f"layers.{b_idx}.attention.q_proj.weight" in k or f"layer.{b_idx}.attention.attention.query.weight" in k)]
        k_k = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.k_proj.weight" in k or f"layers.{b_idx}.attention.k_proj.weight" in k or f"layer.{b_idx}.attention.attention.key.weight" in k)]
        v_k = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.v_proj.weight" in k or f"layers.{b_idx}.attention.v_proj.weight" in k or f"layer.{b_idx}.attention.attention.value.weight" in k)]
        if q_k and k_k and v_k:
            q_w = torch_to_numpy(torch_state[q_k[0]])
            k_w = torch_to_numpy(torch_state[k_k[0]])
            v_w = torch_to_numpy(torch_state[v_k[0]])
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(np.concatenate([q_w, k_w, v_w], axis=0))
            q_b = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.q_proj.bias" in k or f"layers.{b_idx}.attention.q_proj.bias" in k or f"layer.{b_idx}.attention.attention.query.bias" in k)]
            k_b = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.k_proj.bias" in k or f"layers.{b_idx}.attention.k_proj.bias" in k or f"layer.{b_idx}.attention.attention.key.bias" in k)]
            v_b = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.v_proj.bias" in k or f"layers.{b_idx}.attention.v_proj.bias" in k or f"layer.{b_idx}.attention.attention.value.bias" in k)]
            if q_b and k_b and v_b:
                jax_params[f"{prefix}/attn/qkv/bias"] = np.concatenate([torch_to_numpy(torch_state[q_b[0]]), torch_to_numpy(torch_state[k_b[0]]), torch_to_numpy(torch_state[v_b[0]])], axis=0)
        elif any(f"blocks.{b_idx}.attn.qkv.weight" in k for k in torch_state):
            w = torch_to_numpy([torch_state[k] for k in torch_state if f"blocks.{b_idx}.attn.qkv.weight" in k][0])
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(w)
            b_list = [torch_state[k] for k in torch_state if f"blocks.{b_idx}.attn.qkv.bias" in k]
            if b_list:
                jax_params[f"{prefix}/attn/qkv/bias"] = torch_to_numpy(b_list[0])

        out_k = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.o_proj.weight" in k or f"layers.{b_idx}.attention.o_proj.weight" in k or f"layer.{b_idx}.attention.output.dense.weight" in k or f"blocks.{b_idx}.attn.proj.weight" in k)]
        if out_k:
            jax_params[f"{prefix}/attn/out_proj/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[out_k[0]]))
        out_b = [k for k in torch_state if (f"vit.layers.{b_idx}.attention.o_proj.bias" in k or f"layers.{b_idx}.attention.o_proj.bias" in k or f"layer.{b_idx}.attention.output.dense.bias" in k or f"blocks.{b_idx}.attn.proj.bias" in k)]
        if out_b:
            jax_params[f"{prefix}/attn/out_proj/bias"] = torch_to_numpy(torch_state[out_b[0]])

        fc1_k = [k for k in torch_state if (f"vit.layers.{b_idx}.mlp.fc1.weight" in k or f"layers.{b_idx}.mlp.fc1.weight" in k or f"layer.{b_idx}.intermediate.dense.weight" in k or f"blocks.{b_idx}.mlp.fc1.weight" in k)]
        if fc1_k:
            jax_params[f"{prefix}/mlp/fc1/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[fc1_k[0]]))
        fc1_b = [k for k in torch_state if (f"vit.layers.{b_idx}.mlp.fc1.bias" in k or f"layers.{b_idx}.mlp.fc1.bias" in k or f"layer.{b_idx}.intermediate.dense.bias" in k or f"blocks.{b_idx}.mlp.fc1.bias" in k)]
        if fc1_b:
            jax_params[f"{prefix}/mlp/fc1/bias"] = torch_to_numpy(torch_state[fc1_b[0]])

        fc2_k = [k for k in torch_state if (f"vit.layers.{b_idx}.mlp.fc2.weight" in k or f"layers.{b_idx}.mlp.fc2.weight" in k or f"layer.{b_idx}.output.dense.weight" in k or f"blocks.{b_idx}.mlp.fc2.weight" in k)]
        if fc2_k:
            jax_params[f"{prefix}/mlp/fc2/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[fc2_k[0]]))
        fc2_b = [k for k in torch_state if (f"vit.layers.{b_idx}.mlp.fc2.bias" in k or f"layers.{b_idx}.mlp.fc2.bias" in k or f"layer.{b_idx}.output.dense.bias" in k or f"blocks.{b_idx}.mlp.fc2.bias" in k)]
        if fc2_b:
            jax_params[f"{prefix}/mlp/fc2/bias"] = torch_to_numpy(torch_state[fc2_b[0]])

    # Decoder transformer blocks
    dec_depth = (config.decoder_depth if config else 8)
    for b_idx in range(dec_depth):
        prefix = f"decoder_block_{b_idx}"
        for k in [f"decoder.decoder_layers.{b_idx}.layernorm_before.weight", f"decoder_blocks.{b_idx}.norm1.weight"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm1/scale"] = torch_to_numpy(torch_state[k])
                break
        for k in [f"decoder.decoder_layers.{b_idx}.layernorm_before.bias", f"decoder_blocks.{b_idx}.norm1.bias"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm1/bias"] = torch_to_numpy(torch_state[k])
                break
        for k in [f"decoder.decoder_layers.{b_idx}.layernorm_after.weight", f"decoder_blocks.{b_idx}.norm2.weight"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm2/scale"] = torch_to_numpy(torch_state[k])
                break
        for k in [f"decoder.decoder_layers.{b_idx}.layernorm_after.bias", f"decoder_blocks.{b_idx}.norm2.bias"]:
            if k in torch_state:
                jax_params[f"{prefix}/norm2/bias"] = torch_to_numpy(torch_state[k])
                break

        q_k = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.q_proj.weight" in k or f"decoder_layers.{b_idx}.attention.q_proj.weight" in k)]
        k_k = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.k_proj.weight" in k or f"decoder_layers.{b_idx}.attention.k_proj.weight" in k)]
        v_k = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.v_proj.weight" in k or f"decoder_layers.{b_idx}.attention.v_proj.weight" in k)]
        if q_k and k_k and v_k:
            q_w = torch_to_numpy(torch_state[q_k[0]])
            k_w = torch_to_numpy(torch_state[k_k[0]])
            v_w = torch_to_numpy(torch_state[v_k[0]])
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(np.concatenate([q_w, k_w, v_w], axis=0))
            q_b = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.q_proj.bias" in k or f"decoder_layers.{b_idx}.attention.q_proj.bias" in k)]
            k_b = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.k_proj.bias" in k or f"decoder_layers.{b_idx}.attention.k_proj.bias" in k)]
            v_b = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.v_proj.bias" in k or f"decoder_layers.{b_idx}.attention.v_proj.bias" in k)]
            if q_b and k_b and v_b:
                jax_params[f"{prefix}/attn/qkv/bias"] = np.concatenate([torch_to_numpy(torch_state[q_b[0]]), torch_to_numpy(torch_state[k_b[0]]), torch_to_numpy(torch_state[v_b[0]])], axis=0)
        elif any(f"decoder_blocks.{b_idx}.attn.qkv.weight" in k for k in torch_state):
            w = torch_to_numpy([torch_state[k] for k in torch_state if f"decoder_blocks.{b_idx}.attn.qkv.weight" in k][0])
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(w)
            b_list = [torch_state[k] for k in torch_state if f"decoder_blocks.{b_idx}.attn.qkv.bias" in k]
            if b_list:
                jax_params[f"{prefix}/attn/qkv/bias"] = torch_to_numpy(b_list[0])

        out_k = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.o_proj.weight" in k or f"decoder_layers.{b_idx}.attention.o_proj.weight" in k or f"decoder_blocks.{b_idx}.attn.proj.weight" in k)]
        if out_k:
            jax_params[f"{prefix}/attn/out_proj/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[out_k[0]]))
        out_b = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.attention.o_proj.bias" in k or f"decoder_layers.{b_idx}.attention.o_proj.bias" in k or f"decoder_blocks.{b_idx}.attn.proj.bias" in k)]
        if out_b:
            jax_params[f"{prefix}/attn/out_proj/bias"] = torch_to_numpy(torch_state[out_b[0]])

        fc1_k = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.mlp.fc1.weight" in k or f"decoder_layers.{b_idx}.mlp.fc1.weight" in k or f"decoder_blocks.{b_idx}.mlp.fc1.weight" in k)]
        if fc1_k:
            jax_params[f"{prefix}/mlp/fc1/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[fc1_k[0]]))
        fc1_b = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.mlp.fc1.bias" in k or f"decoder_layers.{b_idx}.mlp.fc1.bias" in k or f"decoder_blocks.{b_idx}.mlp.fc1.bias" in k)]
        if fc1_b:
            jax_params[f"{prefix}/mlp/fc1/bias"] = torch_to_numpy(torch_state[fc1_b[0]])

        fc2_k = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.mlp.fc2.weight" in k or f"decoder_layers.{b_idx}.mlp.fc2.weight" in k or f"decoder_blocks.{b_idx}.mlp.fc2.weight" in k)]
        if fc2_k:
            jax_params[f"{prefix}/mlp/fc2/kernel"] = dense_weight_torch_to_jax(torch_to_numpy(torch_state[fc2_k[0]]))
        fc2_b = [k for k in torch_state if (f"decoder.decoder_layers.{b_idx}.mlp.fc2.bias" in k or f"decoder_layers.{b_idx}.mlp.fc2.bias" in k or f"decoder_blocks.{b_idx}.mlp.fc2.bias" in k)]
        if fc2_b:
            jax_params[f"{prefix}/mlp/fc2/bias"] = torch_to_numpy(torch_state[fc2_b[0]])

    nested = unflatten_nested_dict(jax_params, sep="/")
    return {"params": nested}


def convert_torch_efficientnet_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert timm/torchvision EfficientNet PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}
    batch_stats: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        if k.startswith("conv_stem.") or "features.0.0." in k:
            jax_params["conv_stem/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif k.startswith("bn1.") or "features.0.1." in k:
            if "weight" in k:
                jax_params["bn1/scale"] = v_np
            elif "bias" in k:
                jax_params["bn1/bias"] = v_np
            elif "running_mean" in k:
                batch_stats["bn1/mean"] = v_np
            elif "running_var" in k:
                batch_stats["bn1/var"] = v_np

        elif k.startswith("conv_head.") or "features.8.0." in k:
            jax_params["conv_head/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif k.startswith("bn2.") or "features.8.1." in k:
            if "weight" in k:
                jax_params["bn2/scale"] = v_np
            elif "bias" in k:
                jax_params["bn2/bias"] = v_np
            elif "running_mean" in k:
                batch_stats["bn2/mean"] = v_np
            elif "running_var" in k:
                batch_stats["bn2/var"] = v_np

        elif k.startswith("classifier.") or "classifier.1." in k:
            if "weight" in k:
                jax_params["classifier/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "bias" in k:
                jax_params["classifier/bias"] = v_np

        # Stage blocks
        elif "blocks." in k:
            parts = k.split(".")
            # Format: blocks.stage_i.block_j...
            stage_idx = int(parts[1])
            block_idx = int(parts[2])

            d_mult = config.depth_coefficient if config else 1.2
            base_repeats = [1, 2, 2, 3, 3, 4, 1]
            stage_repeats = [math.ceil(d_mult * r) for r in base_repeats]
            flat_idx = sum(stage_repeats[:stage_idx]) + block_idx
            mod_prefix = f"blocks_{flat_idx}"

            is_stage_0 = (stage_idx == 0)

            if is_stage_0:
                if "conv_dw.weight" in k:
                    jax_params[f"{mod_prefix}/depthwise_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "bn1.weight" in k:
                    jax_params[f"{mod_prefix}/depthwise_bn/scale"] = v_np
                elif "bn1.bias" in k:
                    jax_params[f"{mod_prefix}/depthwise_bn/bias"] = v_np
                elif "bn1.running_mean" in k:
                    batch_stats[f"{mod_prefix}/depthwise_bn/mean"] = v_np
                elif "bn1.running_var" in k:
                    batch_stats[f"{mod_prefix}/depthwise_bn/var"] = v_np
                elif "conv_pw.weight" in k:
                    jax_params[f"{mod_prefix}/project_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "bn2.weight" in k:
                    jax_params[f"{mod_prefix}/project_bn/scale"] = v_np
                elif "bn2.bias" in k:
                    jax_params[f"{mod_prefix}/project_bn/bias"] = v_np
                elif "bn2.running_mean" in k:
                    batch_stats[f"{mod_prefix}/project_bn/mean"] = v_np
                elif "bn2.running_var" in k:
                    batch_stats[f"{mod_prefix}/project_bn/var"] = v_np
            else:
                if "conv_pw.weight" in k:
                    jax_params[f"{mod_prefix}/expand_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "bn1.weight" in k:
                    jax_params[f"{mod_prefix}/expand_bn/scale"] = v_np
                elif "bn1.bias" in k:
                    jax_params[f"{mod_prefix}/expand_bn/bias"] = v_np
                elif "bn1.running_mean" in k:
                    batch_stats[f"{mod_prefix}/expand_bn/mean"] = v_np
                elif "bn1.running_var" in k:
                    batch_stats[f"{mod_prefix}/expand_bn/var"] = v_np
                elif "conv_dw.weight" in k:
                    jax_params[f"{mod_prefix}/depthwise_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "bn2.weight" in k:
                    jax_params[f"{mod_prefix}/depthwise_bn/scale"] = v_np
                elif "bn2.bias" in k:
                    jax_params[f"{mod_prefix}/depthwise_bn/bias"] = v_np
                elif "bn2.running_mean" in k:
                    batch_stats[f"{mod_prefix}/depthwise_bn/mean"] = v_np
                elif "bn2.running_var" in k:
                    batch_stats[f"{mod_prefix}/depthwise_bn/var"] = v_np
                elif "conv_pwl.weight" in k:
                    jax_params[f"{mod_prefix}/project_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "bn3.weight" in k:
                    jax_params[f"{mod_prefix}/project_bn/scale"] = v_np
                elif "bn3.bias" in k:
                    jax_params[f"{mod_prefix}/project_bn/bias"] = v_np
                elif "bn3.running_mean" in k:
                    batch_stats[f"{mod_prefix}/project_bn/mean"] = v_np
                elif "bn3.running_var" in k:
                    batch_stats[f"{mod_prefix}/project_bn/var"] = v_np

            # Squeeze-and-Excitation
            if "se.conv_reduce.weight" in k:
                jax_params[f"{mod_prefix}/se/reduce/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "se.conv_reduce.bias" in k:
                jax_params[f"{mod_prefix}/se/reduce/bias"] = v_np
            elif "se.conv_expand.weight" in k:
                jax_params[f"{mod_prefix}/se/expand/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "se.conv_expand.bias" in k:
                jax_params[f"{mod_prefix}/se/expand/bias"] = v_np

    params_nested = unflatten_nested_dict(jax_params, sep="/")
    stats_nested = unflatten_nested_dict(batch_stats, sep="/")
    return {"params": params_nested, "batch_stats": stats_nested}


def convert_torch_clip_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert OpenAI/HuggingFace CLIP PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        if "logit_scale" in k:
            jax_params["logit_scale"] = v_np.reshape(())

        # Visual
        elif "visual.conv1.weight" in k or "vision_model.embeddings.patch_embedding.weight" in k:
            jax_params["visual/conv1/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "visual.class_embedding" in k or "vision_model.embeddings.class_embedding" in k:
            jax_params["visual/class_embedding"] = v_np
        elif "visual.positional_embedding" in k or "vision_model.embeddings.position_embedding.weight" in k:
            jax_params["visual/positional_embedding"] = v_np
        elif "visual.ln_pre.weight" in k or "vision_model.pre_layrnorm.weight" in k:
            jax_params["visual/ln_pre/scale"] = v_np
        elif "visual.ln_pre.bias" in k or "vision_model.pre_layrnorm.bias" in k:
            jax_params["visual/ln_pre/bias"] = v_np
        elif "visual.ln_post.weight" in k or "vision_model.post_layernorm.weight" in k:
            jax_params["visual/ln_post/scale"] = v_np
        elif "visual.ln_post.bias" in k or "vision_model.post_layernorm.bias" in k:
            jax_params["visual/ln_post/bias"] = v_np
        elif "visual.proj" in k or "visual_projection.weight" in k:
            if "visual_projection.weight" in k:
                jax_params["visual/proj/kernel"] = dense_weight_torch_to_jax(v_np)
            else:
                jax_params["visual/proj/kernel"] = v_np

        # Text
        elif "token_embedding.weight" in k:
            jax_params["text/token_embedding/embedding"] = v_np
        elif "text.positional_embedding" in k or k == "positional_embedding" or "text_model.embeddings.position_embedding.weight" in k:
            jax_params["text/positional_embedding"] = v_np
        elif "text_projection" in k:
            if "text_projection.weight" in k:
                jax_params["text/text_projection/kernel"] = dense_weight_torch_to_jax(v_np)
            else:
                jax_params["text/text_projection/kernel"] = v_np
        elif "ln_final.weight" in k or "text_model.final_layer_norm.weight" in k:
            jax_params["text/ln_final/scale"] = v_np
        elif "ln_final.bias" in k or "text_model.final_layer_norm.bias" in k:
            jax_params["text/ln_final/bias"] = v_np

        # Visual Transformer ResBlocks (OpenAI or HF format)
        elif "visual.transformer.resblocks." in k or "vision_model.encoder.layers." in k:
            parts = k.split(".")
            idx_pos = 3 if "visual" in parts[0] else 3
            b_idx = parts[idx_pos]
            prefix = f"visual/resblocks_{b_idx}"

            if "ln_1.weight" in k or "layer_norm1.weight" in k:
                jax_params[f"{prefix}/ln_1/scale"] = v_np
            elif "ln_1.bias" in k or "layer_norm1.bias" in k:
                jax_params[f"{prefix}/ln_1/bias"] = v_np
            elif "attn.in_proj_weight" in k:
                q, k_w, v_w = np.split(v_np, 3, axis=0)
                jax_params[f"{prefix}/attn/q_proj/kernel"] = dense_weight_torch_to_jax(q)
                jax_params[f"{prefix}/attn/k_proj/kernel"] = dense_weight_torch_to_jax(k_w)
                jax_params[f"{prefix}/attn/v_proj/kernel"] = dense_weight_torch_to_jax(v_w)
            elif "attn.in_proj_bias" in k:
                q, k_b, v_b = np.split(v_np, 3, axis=0)
                jax_params[f"{prefix}/attn/q_proj/bias"] = q
                jax_params[f"{prefix}/attn/k_proj/bias"] = k_b
                jax_params[f"{prefix}/attn/v_proj/bias"] = v_b
            elif "attn.out_proj.weight" in k or "self_attn.out_proj.weight" in k:
                jax_params[f"{prefix}/attn/out_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "attn.out_proj.bias" in k or "self_attn.out_proj.bias" in k:
                jax_params[f"{prefix}/attn/out_proj/bias"] = v_np
            elif "self_attn.q_proj.weight" in k:
                jax_params[f"{prefix}/attn/q_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "self_attn.q_proj.bias" in k:
                jax_params[f"{prefix}/attn/q_proj/bias"] = v_np
            elif "self_attn.k_proj.weight" in k:
                jax_params[f"{prefix}/attn/k_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "self_attn.k_proj.bias" in k:
                jax_params[f"{prefix}/attn/k_proj/bias"] = v_np
            elif "self_attn.v_proj.weight" in k:
                jax_params[f"{prefix}/attn/v_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "self_attn.v_proj.bias" in k:
                jax_params[f"{prefix}/attn/v_proj/bias"] = v_np
            elif "ln_2.weight" in k or "layer_norm2.weight" in k:
                jax_params[f"{prefix}/ln_2/scale"] = v_np
            elif "ln_2.bias" in k or "layer_norm2.bias" in k:
                jax_params[f"{prefix}/ln_2/bias"] = v_np
            elif "mlp.c_fc.weight" in k or "mlp.fc1.weight" in k:
                jax_params[f"{prefix}/mlp/c_fc/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "mlp.c_fc.bias" in k or "mlp.fc1.bias" in k:
                jax_params[f"{prefix}/mlp/c_fc/bias"] = v_np
            elif "mlp.c_proj.weight" in k or "mlp.fc2.weight" in k:
                jax_params[f"{prefix}/mlp/c_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "mlp.c_proj.bias" in k or "mlp.fc2.bias" in k:
                jax_params[f"{prefix}/mlp/c_proj/bias"] = v_np

        # Text Transformer ResBlocks (OpenAI or HF format)
        elif ("transformer.resblocks." in k and "visual" not in k) or "text_model.encoder.layers." in k:
            parts = k.split(".")
            idx_pos = 2 if "transformer" in parts[0] else 3
            b_idx = parts[idx_pos]
            prefix = f"text/resblocks_{b_idx}"

            if "ln_1.weight" in k or "layer_norm1.weight" in k:
                jax_params[f"{prefix}/ln_1/scale"] = v_np
            elif "ln_1.bias" in k or "layer_norm1.bias" in k:
                jax_params[f"{prefix}/ln_1/bias"] = v_np
            elif "attn.in_proj_weight" in k:
                q, k_w, v_w = np.split(v_np, 3, axis=0)
                jax_params[f"{prefix}/attn/q_proj/kernel"] = dense_weight_torch_to_jax(q)
                jax_params[f"{prefix}/attn/k_proj/kernel"] = dense_weight_torch_to_jax(k_w)
                jax_params[f"{prefix}/attn/v_proj/kernel"] = dense_weight_torch_to_jax(v_w)
            elif "attn.in_proj_bias" in k:
                q, k_b, v_b = np.split(v_np, 3, axis=0)
                jax_params[f"{prefix}/attn/q_proj/bias"] = q
                jax_params[f"{prefix}/attn/k_proj/bias"] = k_b
                jax_params[f"{prefix}/attn/v_proj/bias"] = v_b
            elif "attn.out_proj.weight" in k or "self_attn.out_proj.weight" in k:
                jax_params[f"{prefix}/attn/out_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "attn.out_proj.bias" in k or "self_attn.out_proj.bias" in k:
                jax_params[f"{prefix}/attn/out_proj/bias"] = v_np
            elif "self_attn.q_proj.weight" in k:
                jax_params[f"{prefix}/attn/q_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "self_attn.q_proj.bias" in k:
                jax_params[f"{prefix}/attn/q_proj/bias"] = v_np
            elif "self_attn.k_proj.weight" in k:
                jax_params[f"{prefix}/attn/k_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "self_attn.k_proj.bias" in k:
                jax_params[f"{prefix}/attn/k_proj/bias"] = v_np
            elif "self_attn.v_proj.weight" in k:
                jax_params[f"{prefix}/attn/v_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "self_attn.v_proj.bias" in k:
                jax_params[f"{prefix}/attn/v_proj/bias"] = v_np
            elif "ln_2.weight" in k or "layer_norm2.weight" in k:
                jax_params[f"{prefix}/ln_2/scale"] = v_np
            elif "ln_2.bias" in k or "layer_norm2.bias" in k:
                jax_params[f"{prefix}/ln_2/bias"] = v_np
            elif "mlp.c_fc.weight" in k or "mlp.fc1.weight" in k:
                jax_params[f"{prefix}/mlp/c_fc/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "mlp.c_fc.bias" in k or "mlp.fc1.bias" in k:
                jax_params[f"{prefix}/mlp/c_fc/bias"] = v_np
            elif "mlp.c_proj.weight" in k or "mlp.fc2.weight" in k:
                jax_params[f"{prefix}/mlp/c_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "mlp.c_proj.bias" in k or "mlp.fc2.bias" in k:
                jax_params[f"{prefix}/mlp/c_proj/bias"] = v_np

    nested = unflatten_nested_dict(jax_params, sep="/")
    return {"params": nested}


def convert_torch_vae_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert diffusers AutoencoderKL PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        # quant_conv & post_quant_conv (check post_quant_conv first to avoid substring collision)
        if "post_quant_conv.weight" in k:
            jax_params["post_quant_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "post_quant_conv.bias" in k:
            jax_params["post_quant_conv/bias"] = v_np
        elif "quant_conv.weight" in k:
            jax_params["quant_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "quant_conv.bias" in k:
            jax_params["quant_conv/bias"] = v_np

        # Encoder stem and output
        elif "encoder.conv_in.weight" in k:
            jax_params["encoder/conv_in/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "encoder.conv_in.bias" in k:
            jax_params["encoder/conv_in/bias"] = v_np
        elif "encoder.conv_out.weight" in k:
            jax_params["encoder/conv_out/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "encoder.conv_out.bias" in k:
            jax_params["encoder/conv_out/bias"] = v_np
        elif "encoder.conv_norm_out.weight" in k:
            jax_params["encoder/conv_norm_out/scale"] = v_np
        elif "encoder.conv_norm_out.bias" in k:
            jax_params["encoder/conv_norm_out/bias"] = v_np

        # Decoder stem and output
        elif "decoder.conv_in.weight" in k:
            jax_params["decoder/conv_in/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "decoder.conv_in.bias" in k:
            jax_params["decoder/conv_in/bias"] = v_np
        elif "decoder.conv_out.weight" in k:
            jax_params["decoder/conv_out/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "decoder.conv_out.bias" in k:
            jax_params["decoder/conv_out/bias"] = v_np
        elif "decoder.conv_norm_out.weight" in k:
            jax_params["decoder/conv_norm_out/scale"] = v_np
        elif "decoder.conv_norm_out.bias" in k:
            jax_params["decoder/conv_norm_out/bias"] = v_np

        # Resnet blocks in Encoder and Decoder
        elif ".resnets." in k:
            parts = k.split(".")
            # Format: {enc_or_dec}.{stage_name}.{stage_idx}.resnets.{res_idx}...
            # or: {enc_or_dec}.mid_block.resnets.{res_idx}...
            root = parts[0]  # "encoder" or "decoder"
            if parts[1] == "mid_block":
                res_idx = parts[3]
                prefix = f"{root}/mid_block_resnets_{res_idx}"
            else:
                stage_name = parts[1]  # "down_blocks" or "up_blocks"
                stage_idx = parts[2]
                res_idx = parts[4]
                prefix = f"{root}/{stage_name}_{stage_idx}_resnets_{res_idx}"

            if "norm1.weight" in k:
                jax_params[f"{prefix}/norm1/scale"] = v_np
            elif "norm1.bias" in k:
                jax_params[f"{prefix}/norm1/bias"] = v_np
            elif "conv1.weight" in k:
                jax_params[f"{prefix}/conv1/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv1.bias" in k:
                jax_params[f"{prefix}/conv1/bias"] = v_np
            elif "norm2.weight" in k:
                jax_params[f"{prefix}/norm2/scale"] = v_np
            elif "norm2.bias" in k:
                jax_params[f"{prefix}/norm2/bias"] = v_np
            elif "conv2.weight" in k:
                jax_params[f"{prefix}/conv2/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv2.bias" in k:
                jax_params[f"{prefix}/conv2/bias"] = v_np
            elif "conv_shortcut.weight" in k:
                jax_params[f"{prefix}/conv_shortcut/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv_shortcut.bias" in k:
                jax_params[f"{prefix}/conv_shortcut/bias"] = v_np

        # Mid block attentions in Encoder and Decoder
        elif ".mid_block.attentions." in k:
            parts = k.split(".")
            root = parts[0]  # "encoder" or "decoder"
            prefix = f"{root}/mid_block_attentions_0"
            if "group_norm.weight" in k:
                jax_params[f"{prefix}/group_norm/scale"] = v_np
            elif "group_norm.bias" in k:
                jax_params[f"{prefix}/group_norm/bias"] = v_np
            elif "to_q.weight" in k:
                jax_params[f"{prefix}/to_q/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "to_q.bias" in k:
                jax_params[f"{prefix}/to_q/bias"] = v_np
            elif "to_k.weight" in k:
                jax_params[f"{prefix}/to_k/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "to_k.bias" in k:
                jax_params[f"{prefix}/to_k/bias"] = v_np
            elif "to_v.weight" in k:
                jax_params[f"{prefix}/to_v/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "to_v.bias" in k:
                jax_params[f"{prefix}/to_v/bias"] = v_np
            elif "to_out.0.weight" in k:
                jax_params[f"{prefix}/to_out_0/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "to_out.0.bias" in k:
                jax_params[f"{prefix}/to_out_0/bias"] = v_np

        # Downsamplers in Encoder
        elif "encoder.down_blocks." in k and ".downsamplers." in k:
            parts = k.split(".")
            stage_idx = parts[2]
            ds_idx = parts[4]
            prefix = f"encoder/down_blocks_{stage_idx}_downsamplers_{ds_idx}"
            if "conv.weight" in k:
                jax_params[f"{prefix}/conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv.bias" in k:
                jax_params[f"{prefix}/conv/bias"] = v_np

        # Upsamplers in Decoder
        elif "decoder.up_blocks." in k and ".upsamplers." in k:
            parts = k.split(".")
            stage_idx = parts[2]
            up_idx = parts[4]
            prefix = f"decoder/up_blocks_{stage_idx}_upsamplers_{up_idx}"
            if "conv.weight" in k:
                jax_params[f"{prefix}/conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv.bias" in k:
                jax_params[f"{prefix}/conv/bias"] = v_np

    nested = unflatten_nested_dict(jax_params, sep="/")
    return {"params": nested}


def convert_torch_unet_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert diffusers UNet2DConditionModel PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        if "conv_in.weight" in k:
            jax_params["conv_in/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "conv_in.bias" in k:
            jax_params["conv_in/bias"] = v_np
        elif "conv_out.weight" in k:
            jax_params["conv_out/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "conv_out.bias" in k:
            jax_params["conv_out/bias"] = v_np
        elif "time_embedding.linear_1.weight" in k:
            jax_params["time_embedding/linear_1/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "time_embedding.linear_1.bias" in k:
            jax_params["time_embedding/linear_1/bias"] = v_np
        elif "time_embedding.linear_2.weight" in k:
            jax_params["time_embedding/linear_2/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "time_embedding.linear_2.bias" in k:
            jax_params["time_embedding/linear_2/bias"] = v_np
        elif "conv_norm_out.weight" in k:
            jax_params["conv_norm_out/scale"] = v_np
        elif "conv_norm_out.bias" in k:
            jax_params["conv_norm_out/bias"] = v_np

        # Resnet blocks in UNet
        elif ".resnets." in k:
            parts = k.split(".")
            # Format: down_blocks.{i}.resnets.{j}... or mid_block.resnets.{j}... or up_blocks.{i}.resnets.{j}...
            if parts[0] == "mid_block":
                prefix = f"mid_block_resnets_{parts[2]}"
            else:
                prefix = f"{parts[0]}_{parts[1]}_resnets_{parts[3]}"

            if "norm1.weight" in k:
                jax_params[f"{prefix}/norm1/scale"] = v_np
            elif "norm1.bias" in k:
                jax_params[f"{prefix}/norm1/bias"] = v_np
            elif "conv1.weight" in k:
                jax_params[f"{prefix}/conv1/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv1.bias" in k:
                jax_params[f"{prefix}/conv1/bias"] = v_np
            elif "time_emb_proj.weight" in k:
                jax_params[f"{prefix}/time_emb_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "time_emb_proj.bias" in k:
                jax_params[f"{prefix}/time_emb_proj/bias"] = v_np
            elif "norm2.weight" in k:
                jax_params[f"{prefix}/norm2/scale"] = v_np
            elif "norm2.bias" in k:
                jax_params[f"{prefix}/norm2/bias"] = v_np
            elif "conv2.weight" in k:
                jax_params[f"{prefix}/conv2/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv2.bias" in k:
                jax_params[f"{prefix}/conv2/bias"] = v_np
            elif "conv_shortcut.weight" in k:
                jax_params[f"{prefix}/conv_shortcut/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv_shortcut.bias" in k:
                jax_params[f"{prefix}/conv_shortcut/bias"] = v_np

        # Attention blocks in UNet
        elif ".attentions." in k:
            parts = k.split(".")
            if parts[0] == "mid_block":
                prefix = "mid_block_attentions_0"
            else:
                prefix = f"{parts[0]}_{parts[1]}_attentions_{parts[3]}"

            if "norm.weight" in k and "transformer_blocks" not in k:
                jax_params[f"{prefix}/norm/scale"] = v_np
            elif "norm.bias" in k and "transformer_blocks" not in k:
                jax_params[f"{prefix}/norm/bias"] = v_np
            elif "proj_in.weight" in k:
                jax_params[f"{prefix}/proj_in/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "proj_in.bias" in k:
                jax_params[f"{prefix}/proj_in/bias"] = v_np
            elif "proj_out.weight" in k:
                jax_params[f"{prefix}/proj_out/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "proj_out.bias" in k:
                jax_params[f"{prefix}/proj_out/bias"] = v_np
            elif "transformer_blocks.0." in k:
                tb = prefix
                if "norm1.weight" in k: jax_params[f"{tb}/norm1/scale"] = v_np
                elif "norm1.bias" in k: jax_params[f"{tb}/norm1/bias"] = v_np
                elif "attn1.to_q.weight" in k: jax_params[f"{tb}/attn1/to_q/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn1.to_k.weight" in k: jax_params[f"{tb}/attn1/to_k/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn1.to_v.weight" in k: jax_params[f"{tb}/attn1/to_v/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn1.to_out.0.weight" in k: jax_params[f"{tb}/attn1/to_out_0/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn1.to_out.0.bias" in k: jax_params[f"{tb}/attn1/to_out_0/bias"] = v_np
                elif "norm2.weight" in k: jax_params[f"{tb}/norm2/scale"] = v_np
                elif "norm2.bias" in k: jax_params[f"{tb}/norm2/bias"] = v_np
                elif "attn2.to_q.weight" in k: jax_params[f"{tb}/attn2/to_q/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn2.to_k.weight" in k: jax_params[f"{tb}/attn2/to_k/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn2.to_v.weight" in k: jax_params[f"{tb}/attn2/to_v/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn2.to_out.0.weight" in k: jax_params[f"{tb}/attn2/to_out_0/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "attn2.to_out.0.bias" in k: jax_params[f"{tb}/attn2/to_out_0/bias"] = v_np
                elif "norm3.weight" in k: jax_params[f"{tb}/norm3/scale"] = v_np
                elif "norm3.bias" in k: jax_params[f"{tb}/norm3/bias"] = v_np
                elif "ff.net.0.proj.weight" in k: jax_params[f"{tb}/ff_net_0_proj/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "ff.net.0.proj.bias" in k: jax_params[f"{tb}/ff_net_0_proj/bias"] = v_np
                elif "ff.net.2.weight" in k: jax_params[f"{tb}/ff_net_2/kernel"] = dense_weight_torch_to_jax(v_np)
                elif "ff.net.2.bias" in k: jax_params[f"{tb}/ff_net_2/bias"] = v_np

        # Downsamplers in UNet
        elif ".downsamplers." in k:
            parts = k.split(".")
            prefix = f"{parts[0]}_{parts[1]}_downsamplers_{parts[3]}"
            if "conv.weight" in k:
                jax_params[f"{prefix}/conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv.bias" in k:
                jax_params[f"{prefix}/conv/bias"] = v_np

        # Upsamplers in UNet
        elif ".upsamplers." in k:
            parts = k.split(".")
            prefix = f"{parts[0]}_{parts[1]}_upsamplers_{parts[3]}"
            if "conv.weight" in k:
                jax_params[f"{prefix}/conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "conv.bias" in k:
                jax_params[f"{prefix}/conv/bias"] = v_np

    nested = unflatten_nested_dict(jax_params, sep="/")
    return {"params": nested}


def convert_torch_diffusion_diff_to_jax(
    torch_state: Dict[str, Any],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Convert sid_unet DiffusionDiff PyTorch state dict to Flax Linen parameters."""
    jax_params: Dict[str, Any] = {}
    batch_stats: Dict[str, Any] = {}

    for k, v in torch_state.items():
        v_np = torch_to_numpy(v)

        # Normalization of Z
        if "z_norm.weight" in k:
            jax_params["z_norm/scale"] = v_np
        elif "z_norm.bias" in k:
            jax_params["z_norm/bias"] = v_np

        # Decoder initial conv
        elif "decoder.init_conv.0.weight" in k:
            jax_params["decoder/init_conv_0/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "decoder.init_conv.1.weight" in k:
            jax_params["decoder/init_conv_1/scale"] = v_np
        elif "decoder.init_conv.1.bias" in k:
            jax_params["decoder/init_conv_1/bias"] = v_np
        elif "decoder.init_conv.1.running_mean" in k:
            batch_stats["decoder/init_conv_1/mean"] = v_np
        elif "decoder.init_conv.1.running_var" in k:
            batch_stats["decoder/init_conv_1/var"] = v_np

        # Decoder stages upsampling conv
        elif "decoder.stages." in k:
            parts = k.split(".")
            stage_idx = parts[2]
            stage_prefix = f"decoder/stages_{stage_idx}"

            if "up.1.weight" in k or "up_conv.weight" in k:
                jax_params[f"{stage_prefix}/up_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "norm.weight" in k and "res_blocks" not in k:
                jax_params[f"{stage_prefix}/norm/scale"] = v_np
            elif "norm.bias" in k and "res_blocks" not in k:
                jax_params[f"{stage_prefix}/norm/bias"] = v_np
            elif "norm.running_mean" in k and "res_blocks" not in k:
                batch_stats[f"{stage_prefix}/norm/mean"] = v_np
            elif "norm.running_var" in k and "res_blocks" not in k:
                batch_stats[f"{stage_prefix}/norm/var"] = v_np

            elif "res_blocks." in k:
                res_idx = parts[4]
                res_prefix = f"{stage_prefix}/res_blocks_{res_idx}"
                if "conv1.weight" in k:
                    jax_params[f"{res_prefix}/conv1/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "norm1.weight" in k:
                    jax_params[f"{res_prefix}/norm1/scale"] = v_np
                elif "norm1.bias" in k:
                    jax_params[f"{res_prefix}/norm1/bias"] = v_np
                elif "norm1.running_mean" in k:
                    batch_stats[f"{res_prefix}/norm1/mean"] = v_np
                elif "norm1.running_var" in k:
                    batch_stats[f"{res_prefix}/norm1/var"] = v_np
                elif "conv2.weight" in k:
                    jax_params[f"{res_prefix}/conv2/kernel"] = conv2d_weight_torch_to_jax(v_np)
                elif "norm2.weight" in k:
                    jax_params[f"{res_prefix}/norm2/scale"] = v_np
                elif "norm2.bias" in k:
                    jax_params[f"{res_prefix}/norm2/bias"] = v_np
                elif "norm2.running_mean" in k:
                    batch_stats[f"{res_prefix}/norm2/mean"] = v_np
                elif "norm2.running_var" in k:
                    batch_stats[f"{res_prefix}/norm2/var"] = v_np

        # Skip fusions
        elif "decoder.skip_fusions." in k:
            parts = k.split(".")
            f_idx = parts[2]
            f_prefix = f"decoder/skip_fusions_{f_idx}"
            if "conv.weight" in k:
                jax_params[f"{f_prefix}/conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "norm.weight" in k:
                jax_params[f"{f_prefix}/norm/scale"] = v_np
            elif "norm.bias" in k:
                jax_params[f"{f_prefix}/norm/bias"] = v_np
            elif "skip_norm.weight" in k:
                jax_params[f"{f_prefix}/skip_norm/scale"] = v_np
            elif "skip_norm.bias" in k:
                jax_params[f"{f_prefix}/skip_norm/bias"] = v_np

        # Perpendicular skip fusions (V2 / Minimized)
        elif "decoder.perp_skip_fusions." in k or "perp_skip_fusions." in k:
            parts = k.split(".")
            idx_pos = 2 if "decoder" in parts[0] else 1
            f_idx = parts[idx_pos]
            f_prefix = f"decoder/perp_skip_fusions_{f_idx}"
            if "conv.weight" in k:
                jax_params[f"{f_prefix}/conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "norm.weight" in k:
                jax_params[f"{f_prefix}/norm/scale"] = v_np
            elif "norm.bias" in k:
                jax_params[f"{f_prefix}/norm/bias"] = v_np
            elif "perp_norm.weight" in k:
                jax_params[f"{f_prefix}/perp_norm/scale"] = v_np
            elif "perp_norm.bias" in k:
                jax_params[f"{f_prefix}/perp_norm/bias"] = v_np

        # Output conv
        elif "decoder.out_conv.weight" in k:
            jax_params["decoder/out_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "decoder.out_conv.bias" in k:
            jax_params["decoder/out_conv/bias"] = v_np

        # Classifier head
        elif "classifier_head.fc1.weight" in k:
            jax_params["classifier_head/fc1/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "classifier_head.fc1.bias" in k:
            jax_params["classifier_head/fc1/bias"] = v_np
        elif "classifier_head.fc2.weight" in k:
            jax_params["classifier_head/fc2/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "classifier_head.fc2.bias" in k:
            jax_params["classifier_head/fc2/bias"] = v_np

    params_nested = unflatten_nested_dict(jax_params, sep="/")
    result: Dict[str, Any] = {"params": params_nested}
    if batch_stats:
        result["batch_stats"] = unflatten_nested_dict(batch_stats, sep="/")
    return result


def convert_torch_state_dict_to_jax(
    torch_state: Dict[str, Any],
    model_type: str,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Universal dispatcher for converting PyTorch state dicts to Flax Linen parameters."""
    model_type = model_type.lower()
    if model_type in ("vit", "vit_b", "vit_l", "vision_transformer"):
        return convert_torch_vit_to_jax(torch_state, config)
    elif model_type in ("mae", "masked_autoencoder"):
        return convert_torch_mae_to_jax(torch_state, config)
    elif model_type in ("efficientnet", "efficientnet_b2"):
        return convert_torch_efficientnet_to_jax(torch_state, config)
    elif model_type in ("clip", "clip_vit"):
        return convert_torch_clip_to_jax(torch_state, config)
    elif model_type in ("sd_vae", "vae", "autoencoder_kl"):
        return convert_torch_vae_to_jax(torch_state, config)
    elif model_type in ("sd_unet", "unet"):
        return convert_torch_unet_to_jax(torch_state, config)
    elif model_type in ("diffusion_diff", "diffusion_diff_v2", "diffusion_diff_minimized"):
        return convert_torch_diffusion_diff_to_jax(torch_state, config)
    else:
        raise ValueError(f"Unknown model_type for conversion: {model_type}")
