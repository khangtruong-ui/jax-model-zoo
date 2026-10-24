"""DiffusionDiff models family in Flax Linen (jax.nn)."""

from jax_model_zoo.models.diffusion_diff.blocks import (
    AuxiliaryClassifier,
    DecoderUpsampleStage,
    PerpendicularSkipFusion,
    ResidualConvBlock,
    SkipFusion,
    expand_to_spatial,
    sinusoidal_embedding,
)
from jax_model_zoo.models.diffusion_diff.diffusion_diff import (
    DiffuserNoiseDecoder,
    DiffusionDiff,
    DiffusionDiffModel,
    DiffusionMultiNoiseDecoder,
    TrainableLatentDecoder,
)
from jax_model_zoo.models.diffusion_diff.diffusion_diff_minimized import (
    DiffusionDiffMinimized,
    DiffusionDiffMinimizedModel,
)
from jax_model_zoo.models.diffusion_diff.diffusion_diff_v2 import (
    DiffusionDiffV2,
    DiffusionDiffV2Model,
    TrainableLatentDecoderV2,
)

__all__ = [
    "sinusoidal_embedding",
    "expand_to_spatial",
    "ResidualConvBlock",
    "DecoderUpsampleStage",
    "SkipFusion",
    "PerpendicularSkipFusion",
    "AuxiliaryClassifier",
    "TrainableLatentDecoder",
    "DiffusionDiff",
    "DiffusionDiffModel",
    "DiffuserNoiseDecoder",
    "DiffusionMultiNoiseDecoder",
    "TrainableLatentDecoderV2",
    "DiffusionDiffV2",
    "DiffusionDiffV2Model",
    "DiffusionDiffMinimized",
    "DiffusionDiffMinimizedModel",
]
