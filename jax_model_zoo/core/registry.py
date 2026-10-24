"""Central model registry and online checkpoint downloading system for JAX Model Zoo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from jax_model_zoo.core.checkpoint import save_checkpoint, load_checkpoint


PRETRAINED_CATALOG: Dict[str, Dict[str, Any]] = {
    "vit-b": {
        "aliases": ["vit-base", "vit-base-patch16-224", "google/vit-base-patch16-224"],
        "architecture": "vit_base_patch16_224",
        "online_source": "google/vit-base-patch16-224",
        "source_type": "huggingface",
        "description": "Vision Transformer Base (12 layers, 768 width, 12 heads, patch 16, 224x224)",
        "input_shape": (1, 224, 224, 3),
    },
    "vit-l": {
        "aliases": ["vit-large", "vit-large-patch16-224", "google/vit-large-patch16-224"],
        "architecture": "vit_large_patch16_224",
        "online_source": "google/vit-large-patch16-224",
        "source_type": "huggingface",
        "description": "Vision Transformer Large (24 layers, 1024 width, 16 heads, patch 16, 224x224)",
        "input_shape": (1, 224, 224, 3),
    },
    "mae-base": {
        "aliases": ["mae", "vit-mae-base", "facebook/vit-mae-base"],
        "architecture": "mae_vit_base_patch16",
        "online_source": "facebook/vit-mae-base",
        "source_type": "huggingface",
        "description": "Masked Autoencoder Base (12-layer ViT encoder + 8-layer ViT decoder)",
        "input_shape": (1, 224, 224, 3),
    },
    "mae-large": {
        "aliases": ["vit-mae-large", "facebook/vit-mae-large"],
        "architecture": "mae_vit_large_patch16",
        "online_source": "facebook/vit-mae-large",
        "source_type": "huggingface",
        "description": "Masked Autoencoder Large (24-layer ViT encoder + 8-layer ViT decoder)",
        "input_shape": (1, 224, 224, 3),
    },
    "efficientnet-b2": {
        "aliases": ["efficientnet_b2", "timm/efficientnet_b2.ra_in1k"],
        "architecture": "efficientnet_b2",
        "online_source": "efficientnet_b2",
        "source_type": "timm",
        "description": "EfficientNet-B2 pretrained on ImageNet-1k (288x288 input)",
        "input_shape": (1, 288, 288, 3),
    },
    "clip-vit-base-patch32": {
        "aliases": ["clip", "clip-base-32", "openai/clip-vit-base-patch32"],
        "architecture": "clip_vit_base_patch32",
        "online_source": "openai/clip-vit-base-patch32",
        "source_type": "huggingface",
        "description": "OpenAI CLIP ViT-B/32 multimodal vision-and-text backbone",
        "input_shape": (1, 224, 224, 3),
    },
    "clip-vit-base-patch16": {
        "aliases": ["clip-base-16", "openai/clip-vit-base-patch16"],
        "architecture": "clip_vit_base_patch16",
        "online_source": "openai/clip-vit-base-patch16",
        "source_type": "huggingface",
        "description": "OpenAI CLIP ViT-B/16 multimodal vision-and-text backbone",
        "input_shape": (1, 224, 224, 3),
    },
    "clip-vit-large-patch14": {
        "aliases": ["clip-large-14", "openai/clip-vit-large-patch14"],
        "architecture": "clip_vit_large_patch14",
        "online_source": "openai/clip-vit-large-patch14",
        "source_type": "huggingface",
        "description": "OpenAI CLIP ViT-L/14 multimodal vision-and-text backbone",
        "input_shape": (1, 224, 224, 3),
    },
    "stable-diffusion-v1-5": {
        "aliases": ["sd-1.5", "sd-v1-5", "runwayml/stable-diffusion-v1-5"],
        "architecture": "stable_diffusion_v1_5",
        "online_source": "runwayml/stable-diffusion-v1-5",
        "source_type": "diffusers",
        "description": "RunwayML Stable Diffusion v1.5 (AutoencoderKL VAE + UNet2DConditionModel)",
        "input_shape": (1, 512, 512, 3),
    },
    "stable-diffusion-efficient": {
        "aliases": ["sd-efficient", "bk-sdm-small", "nota-ai/bk-sdm-small"],
        "architecture": "stable_diffusion_efficient",
        "online_source": "nota-ai/bk-sdm-small",
        "source_type": "diffusers",
        "description": "Efficient Block-Pruned Stable Diffusion (BK-SDM-small / Tiny-SD)",
        "input_shape": (1, 512, 512, 3),
    },
    "diffusion-diff": {
        "aliases": ["diffusion_diff", "diffusion-diff-v1", "diffusion-diff-v2", "diffusion-diff-minimized"],
        "architecture": "diffusion_diff",
        "online_source": "https://github.com/khangtruong-ui/ai-detection-unet-gpu",
        "source_type": "repo",
        "description": "DiffusionDiff AI Detection UNet (architectures: standard, v2, minimized)",
        "input_shape": (1, 256, 256, 3),
        "has_pretrained_online": False,
    },
}


def canonicalize_model_name(name: str) -> str:
    """Resolve aliases to the canonical model identifier."""
    name_clean = name.strip().lower()
    for canonical, info in PRETRAINED_CATALOG.items():
        if canonical == name_clean:
            return canonical
        if name_clean in [a.lower() for a in info.get("aliases", [])]:
            return canonical
    return name


def list_models() -> List[str]:
    """List all available canonical model names in the catalog."""
    return list(PRETRAINED_CATALOG.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get metadata for a model in the catalog."""
    canonical = canonicalize_model_name(model_name)
    if canonical not in PRETRAINED_CATALOG:
        raise KeyError(
            f"Model '{model_name}' not found in registry. Available models: {list_models()}"
        )
    return PRETRAINED_CATALOG[canonical]


def get_default_cache_dir() -> Path:
    """Get default directory for cached model weights."""
    cache_root = os.environ.get(
        "JAX_MODEL_ZOO_CACHE",
        os.path.expanduser("~/.cache/jax_model_zoo"),
    )
    path = Path(cache_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model_from_online(
    model_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Directly instantiate the model and load online pretrained weights.
    No local random initialization is performed.
    """
    canonical = canonicalize_model_name(model_name)
    info = get_model_info(canonical)
    arch = info["architecture"]
    source = info["online_source"]

    if canonical in ("vit-b", "vit-l"):
        from jax_model_zoo.models.vision_transformer.vit import VisionTransformer
        return VisionTransformer.from_pretrained(source)

    elif canonical in ("mae-base", "mae-large"):
        from jax_model_zoo.models.vision_transformer.mae import MaskedAutoencoder
        return MaskedAutoencoder.from_pretrained(source)

    elif canonical == "efficientnet-b2":
        from jax_model_zoo.models.efficientnet.efficientnet import EfficientNet
        return EfficientNet.from_pretrained(source)

    elif canonical in ("clip-vit-base-patch32", "clip-vit-base-patch16", "clip-vit-large-patch14"):
        from jax_model_zoo.models.clip.clip import CLIP
        return CLIP.from_pretrained(source)

    elif canonical == "stable-diffusion-v1-5":
        from jax_model_zoo.models.stable_diffusion.pipeline import StableDiffusionPipeline
        return StableDiffusionPipeline.from_pretrained(source, is_efficient=False)

    elif canonical == "stable-diffusion-efficient":
        from jax_model_zoo.models.stable_diffusion.pipeline import StableDiffusionPipeline
        return StableDiffusionPipeline.from_pretrained(source, is_efficient=True)

    elif canonical == "diffusion-diff":
        raise ValueError(
            "DiffusionDiff models currently do not have public online pretrained weights. "
            "Please initialize from DiffusionDiffConfig or supply user checkpoints."
        )

    else:
        raise ValueError(f"Unsupported model architecture: {arch}")


def download_and_convert(
    model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    format: str = "safetensors",
    verify: bool = True,
) -> Tuple[Any, Dict[str, Any], Path]:
    """
    Download pretrained weights from online reputable HuggingFace/official sources,
    convert to topology-agnostic JAX Linen parameters, optionally verify with a
    forward pass, and save to output_dir.
    """
    canonical = canonicalize_model_name(model_name)
    info = get_model_info(canonical)

    out_path = Path(output_dir) if output_dir else get_default_cache_dir()
    out_path.mkdir(parents=True, exist_ok=True)

    file_ext = {
        "safetensors": ".safetensors",
        "msgpack": ".msgpack",
        "orbax": ".orbax",
    }.get(format, ".safetensors")

    save_dest = out_path / f"{canonical}{file_ext}"

    # Load from online
    model, params = load_model_from_online(canonical)

    # Verify forward pass if requested
    if verify:
        import jax.numpy as jnp
        shape = info.get("input_shape", (1, 224, 224, 3))
        dummy_inp = jnp.ones(shape, dtype=jnp.float32)

        if canonical in ("vit-b", "vit-l"):
            _ = model.apply(params, dummy_inp, deterministic=True)
        elif canonical in ("mae-base", "mae-large"):
            _ = model.apply(params, dummy_inp, method=model.extract_features)
        elif canonical == "efficientnet-b2":
            _ = model.apply(params, dummy_inp, deterministic=True)
        elif canonical.startswith("clip"):
            _ = model.apply(params, dummy_inp, method=model.encode_image)

    # Save converted weights
    save_checkpoint(params, save_dest, format=format)
    return model, params, save_dest
