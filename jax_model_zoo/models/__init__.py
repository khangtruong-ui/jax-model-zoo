"""Models zoo in Flax Linen (jax.nn)."""

from jax_model_zoo.models.clip import (
    CLIP,
    CLIPTextTransformer,
    CLIPVisionTransformer,
    clip_vit_base_patch16,
    clip_vit_base_patch32,
    clip_vit_large_patch14,
)
from jax_model_zoo.models.diffusion_diff import (
    DiffusionDiff,
    DiffusionDiffMinimized,
    DiffusionDiffV2,
    TrainableLatentDecoder,
    TrainableLatentDecoderV2,
)
from jax_model_zoo.models.efficientnet import (
    EfficientNet,
    efficientnet_b2,
)
from jax_model_zoo.models.stable_diffusion import (
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    stable_diffusion_efficient,
    stable_diffusion_v1_5,
)
from jax_model_zoo.models.vision_transformer import (
    MaskedAutoencoder,
    VisionTransformer,
    mae_vit_base_patch16,
    mae_vit_large_patch16,
    vit_base_patch16_224,
    vit_large_patch14_224,
    vit_large_patch16_224,
)

__all__ = [
    "VisionTransformer",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "vit_large_patch14_224",
    "MaskedAutoencoder",
    "mae_vit_base_patch16",
    "mae_vit_large_patch16",
    "EfficientNet",
    "efficientnet_b2",
    "CLIP",
    "CLIPVisionTransformer",
    "CLIPTextTransformer",
    "clip_vit_base_patch32",
    "clip_vit_large_patch14",
    "AutoencoderKL",
    "UNet2DConditionModel",
    "StableDiffusionPipeline",
    "stable_diffusion_v1_5",
    "stable_diffusion_efficient",
    "DiffusionDiff",
    "DiffusionDiffV2",
    "DiffusionDiffMinimized",
    "TrainableLatentDecoder",
    "TrainableLatentDecoderV2",
]
