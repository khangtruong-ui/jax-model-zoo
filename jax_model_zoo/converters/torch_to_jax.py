"""Converters from PyTorch checkpoint state_dict to Flax Linen parameter PyTree."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
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
        if "embeddings.patch_embeddings.projection.weight" in k or "patch_embed.proj.weight" in k:
            jax_params["patch_embed/projection/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "embeddings.patch_embeddings.projection.bias" in k or "patch_embed.proj.bias" in k:
            jax_params["patch_embed/projection/bias"] = v_np

        # 2. CLS and Positional embeddings
        elif "embeddings.cls_token" in k or k == "cls_token":
            jax_params["cls_token"] = v_np.reshape((1, 1, -1))
        elif "embeddings.position_embeddings" in k or "pos_embed" in k:
            jax_params["pos_embed"] = v_np.reshape((1, -1, v_np.shape[-1]))

        # 3. Final LayerNorm
        elif "layernorm.weight" in k or k == "norm.weight":
            jax_params["norm/scale"] = v_np
        elif "layernorm.bias" in k or k == "norm.bias":
            jax_params["norm/bias"] = v_np

        # 4. Classifier Head
        elif "classifier.weight" in k or "head.weight" in k:
            jax_params["head/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "classifier.bias" in k or "head.bias" in k:
            jax_params["head/bias"] = v_np

        # 5. Transformer Blocks
        elif "layer." in k or "blocks." in k:
            # Extract block index
            parts = k.split(".")
            block_idx = None
            for idx, p in enumerate(parts):
                if p in ("layer", "blocks") and idx + 1 < len(parts):
                    try:
                        block_idx = int(parts[idx + 1])
                        break
                    except ValueError:
                        pass

            if block_idx is None:
                continue

            prefix = f"block_{block_idx}"

            if "layernorm_before.weight" in k or "norm1.weight" in k:
                jax_params[f"{prefix}/norm1/scale"] = v_np
            elif "layernorm_before.bias" in k or "norm1.bias" in k:
                jax_params[f"{prefix}/norm1/bias"] = v_np

            elif "attention.output.dense.weight" in k or "attn.proj.weight" in k:
                jax_params[f"{prefix}/attn/out_proj/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "attention.output.dense.bias" in k or "attn.proj.bias" in k:
                jax_params[f"{prefix}/attn/out_proj/bias"] = v_np

            elif "layernorm_after.weight" in k or "norm2.weight" in k:
                jax_params[f"{prefix}/norm2/scale"] = v_np
            elif "layernorm_after.bias" in k or "norm2.bias" in k:
                jax_params[f"{prefix}/norm2/bias"] = v_np

            elif "intermediate.dense.weight" in k or "mlp.fc1.weight" in k:
                jax_params[f"{prefix}/mlp/fc1/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "intermediate.dense.bias" in k or "mlp.fc1.bias" in k:
                jax_params[f"{prefix}/mlp/fc1/bias"] = v_np

            elif "output.dense.weight" in k or "mlp.fc2.weight" in k:
                jax_params[f"{prefix}/mlp/fc2/kernel"] = dense_weight_torch_to_jax(v_np)
            elif "output.dense.bias" in k or "mlp.fc2.bias" in k:
                jax_params[f"{prefix}/mlp/fc2/bias"] = v_np

    # Collect q, k, v for attention if separate in torch
    num_blocks = (config.depth if config else 24)
    for b_idx in range(num_blocks):
        prefix = f"block_{b_idx}"
        q_k = [k for k in torch_state if f"layer.{b_idx}.attention.attention.query.weight" in k]
        k_k = [k for k in torch_state if f"layer.{b_idx}.attention.attention.key.weight" in k]
        v_k = [k for k in torch_state if f"layer.{b_idx}.attention.attention.value.weight" in k]

        if q_k and k_k and v_k:
            q_w = torch_to_numpy(torch_state[q_k[0]])
            k_w = torch_to_numpy(torch_state[k_k[0]])
            v_w = torch_to_numpy(torch_state[v_k[0]])
            qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(qkv_w)

            q_b = [k for k in torch_state if f"layer.{b_idx}.attention.attention.query.bias" in k]
            k_b = [k for k in torch_state if f"layer.{b_idx}.attention.attention.key.bias" in k]
            v_b = [k for k in torch_state if f"layer.{b_idx}.attention.attention.value.bias" in k]
            if q_b and k_b and v_b:
                qkv_b = np.concatenate([
                    torch_to_numpy(torch_state[q_b[0]]),
                    torch_to_numpy(torch_state[k_b[0]]),
                    torch_to_numpy(torch_state[v_b[0]]),
                ], axis=0)
                jax_params[f"{prefix}/attn/qkv/bias"] = qkv_b
        elif any(f"blocks.{b_idx}.attn.qkv.weight" in k for k in torch_state):
            # timm format where qkv is already concatenated
            w = torch_to_numpy([torch_state[k] for k in torch_state if f"blocks.{b_idx}.attn.qkv.weight" in k][0])
            jax_params[f"{prefix}/attn/qkv/kernel"] = dense_weight_torch_to_jax(w)
            b_list = [torch_state[k] for k in torch_state if f"blocks.{b_idx}.attn.qkv.bias" in k]
            if b_list:
                jax_params[f"{prefix}/attn/qkv/bias"] = torch_to_numpy(b_list[0])

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

        if "patch_embed.proj.weight" in k or "patch_embed.projection.weight" in k:
            jax_params["patch_embed/projection/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "patch_embed.proj.bias" in k or "patch_embed.projection.bias" in k:
            jax_params["patch_embed/projection/bias"] = v_np
        elif k.endswith("cls_token"):
            jax_params["cls_token"] = v_np.reshape((1, 1, -1))
        elif k.endswith("pos_embed") and "decoder" not in k:
            # If pos_embed contains cls token in torch [1, N+1, D], take the patch portion [1, N, D]
            if v_np.shape[1] != (v_np.shape[1] - 1) and "decoder" not in k and v_np.ndim == 3 and v_np.shape[1] > 196:
                v_np = v_np[:, 1:, :]
            jax_params["pos_embed"] = v_np
        elif k.endswith("mask_token"):
            jax_params["mask_token"] = v_np.reshape((1, 1, -1))
        elif "decoder_pos_embed" in k:
            jax_params["decoder_pos_embed"] = v_np
        elif "decoder_embed.weight" in k:
            jax_params["decoder_embed/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "decoder_embed.bias" in k:
            jax_params["decoder_embed/bias"] = v_np
        elif "decoder_pred.weight" in k:
            jax_params["decoder_pred/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "decoder_pred.bias" in k:
            jax_params["decoder_pred/bias"] = v_np
        elif "norm.weight" in k and "decoder" not in k and "blocks" not in k:
            jax_params["encoder_norm/scale"] = v_np
        elif "norm.bias" in k and "decoder" not in k and "blocks" not in k:
            jax_params["encoder_norm/bias"] = v_np
        elif "decoder_norm.weight" in k:
            jax_params["decoder_norm/scale"] = v_np
        elif "decoder_norm.bias" in k:
            jax_params["decoder_norm/bias"] = v_np

    # Encoder and decoder blocks
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

        if "conv_stem.weight" in k:
            jax_params["conv_stem/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "bn1.weight" in k:
            jax_params["bn1/scale"] = v_np
        elif "bn1.bias" in k:
            jax_params["bn1/bias"] = v_np
        elif "bn1.running_mean" in k:
            batch_stats["bn1/mean"] = v_np
        elif "bn1.running_var" in k:
            batch_stats["bn1/var"] = v_np

        elif "conv_head.weight" in k:
            jax_params["conv_head/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "bn2.weight" in k:
            jax_params["bn2/scale"] = v_np
        elif "bn2.bias" in k:
            jax_params["bn2/bias"] = v_np
        elif "bn2.running_mean" in k:
            batch_stats["bn2/mean"] = v_np
        elif "bn2.running_var" in k:
            batch_stats["bn2/var"] = v_np

        elif "classifier.weight" in k:
            jax_params["classifier/kernel"] = dense_weight_torch_to_jax(v_np)
        elif "classifier.bias" in k:
            jax_params["classifier/bias"] = v_np

        # Stage blocks
        elif "blocks." in k:
            parts = k.split(".")
            # Format: blocks.stage_i.block_j...
            stage_idx = parts[1]
            block_idx = parts[2]
            mod_prefix = f"blocks_{stage_idx}_{block_idx}"

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
            elif "se.conv_reduce.weight" in k:
                jax_params[f"{mod_prefix}/se/reduce/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "se.conv_reduce.bias" in k:
                jax_params[f"{mod_prefix}/se/reduce/bias"] = v_np
            elif "se.conv_expand.weight" in k:
                jax_params[f"{mod_prefix}/se/expand/kernel"] = conv2d_weight_torch_to_jax(v_np)
            elif "se.conv_expand.bias" in k:
                jax_params[f"{mod_prefix}/se/expand/bias"] = v_np
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
        elif "visual.conv1.weight" in k:
            jax_params["visual/conv1/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "visual.class_embedding" in k:
            jax_params["visual/class_embedding"] = v_np
        elif "visual.positional_embedding" in k:
            jax_params["visual/positional_embedding"] = v_np
        elif "visual.ln_pre.weight" in k:
            jax_params["visual/ln_pre/scale"] = v_np
        elif "visual.ln_pre.bias" in k:
            jax_params["visual/ln_pre/bias"] = v_np
        elif "visual.ln_post.weight" in k:
            jax_params["visual/ln_post/scale"] = v_np
        elif "visual.ln_post.bias" in k:
            jax_params["visual/ln_post/bias"] = v_np
        elif "visual.proj" in k:
            jax_params["visual/proj/kernel"] = v_np

        # Text
        elif "token_embedding.weight" in k:
            jax_params["text/token_embedding/embedding"] = v_np
        elif "text.positional_embedding" in k or k == "positional_embedding":
            jax_params["text/positional_embedding"] = v_np
        elif "text_projection" in k:
            jax_params["text/text_projection/kernel"] = v_np
        elif "ln_final.weight" in k:
            jax_params["text/ln_final/scale"] = v_np
        elif "ln_final.bias" in k:
            jax_params["text/ln_final/bias"] = v_np

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

        # quant_conv & post_quant_conv
        if "quant_conv.weight" in k:
            jax_params["quant_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "quant_conv.bias" in k:
            jax_params["quant_conv/bias"] = v_np
        elif "post_quant_conv.weight" in k:
            jax_params["post_quant_conv/kernel"] = conv2d_weight_torch_to_jax(v_np)
        elif "post_quant_conv.bias" in k:
            jax_params["post_quant_conv/bias"] = v_np

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
