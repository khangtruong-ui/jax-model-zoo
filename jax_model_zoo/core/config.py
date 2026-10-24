"""Configuration dataclasses for JAX Model Zoo architectures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml


@dataclass
class BaseModelConfig:
    """Base configuration class with serialization utilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Create configuration from dictionary, ignoring unknown fields."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> Any:
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Any:
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class ViTConfig(BaseModelConfig):
    """Configuration for Vision Transformer (ViT-B, ViT-L, etc.)."""

    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    classifier: str = "token"  # 'token', 'gap', or 'none'
    act_layer: str = "gelu"


@dataclass
class MAEConfig(BaseModelConfig):
    """Configuration for Masked Autoencoder (MAE)."""

    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    decoder_mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True


@dataclass
class EfficientNetConfig(BaseModelConfig):
    """Configuration for EfficientNet models (e.g. EfficientNet-B2)."""

    width_coefficient: float = 1.1
    depth_coefficient: float = 1.2
    image_size: int = 288
    dropout_rate: float = 0.3
    drop_connect_rate: float = 0.2
    in_channels: int = 3
    num_classes: int = 1000
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    act_layer: str = "silu"


@dataclass
class CLIPVisionConfig(BaseModelConfig):
    """Configuration for CLIP Vision Transformer."""

    image_size: int = 224
    patch_size: int = 32
    width: int = 768
    layers: int = 12
    heads: int = 12
    output_dim: int = 512
    mlp_dim: Optional[int] = None
    act_layer: str = "quick_gelu"


@dataclass
class CLIPTextConfig(BaseModelConfig):
    """Configuration for CLIP Text Transformer."""

    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    output_dim: int = 512
    mlp_dim: Optional[int] = None
    act_layer: str = "quick_gelu"
    layers: int = 12
    output_dim: int = 512
    act_layer: str = "quick_gelu"


@dataclass
class CLIPConfig(BaseModelConfig):
    """Configuration for multimodal CLIP."""

    vision: CLIPVisionConfig = field(default_factory=CLIPVisionConfig)
    text: CLIPTextConfig = field(default_factory=CLIPTextConfig)
    logit_scale_init: float = 2.6592  # ln(1/0.07)


@dataclass
class AutoencoderKLConfig(BaseModelConfig):
    """Configuration for Stable Diffusion VAE (AutoencoderKL)."""

    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 512
    scaling_factor: float = 0.18215


@dataclass
class UNet2DConditionConfig(BaseModelConfig):
    """Configuration for UNet2DConditionModel in Stable Diffusion."""

    sample_size: int = 64
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    )
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    cross_attention_dim: int = 768
    act_fn: str = "silu"
    norm_num_groups: int = 32
    num_class_embeds: Optional[int] = None


@dataclass
class StableDiffusionConfig(BaseModelConfig):
    """Configuration for Stable Diffusion Pipeline components."""

    vae: AutoencoderKLConfig = field(default_factory=AutoencoderKLConfig)
    unet: UNet2DConditionConfig = field(default_factory=UNet2DConditionConfig)
    text_encoder: CLIPTextConfig = field(default_factory=CLIPTextConfig)
    is_efficient: bool = False


@dataclass
class DiffusionDiffConfig(BaseModelConfig):
    """Configuration for DiffusionDiff models family."""

    version: str = "v1"  # "v1", "v2", or "minimized"
    timesteps: List[int] = field(default_factory=lambda: [100, 500, 900])
    timestep_embed_dim: int = 32
    sigma_embed_dim: int = 32
    include_z0: bool = True
    include_noisy_latents: bool = True
    include_added_noise: bool = True
    include_predicted_noise: bool = True
    include_noise_diff: bool = True
    scaling_factor: float = 0.18215
    use_skip_connections: bool = True
    skip_channels: Optional[List[int]] = None
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    out_channels: int = 1
    decoder_res_blocks: int = 1
    decoder_norm: str = "batchnorm"
    decoder_act: str = "silu"
    decoder_dropout: float = 0.0
    use_fpn: bool = False
    fpn_channels: int = 128
    aux_classifier: bool = False
    num_classes: int = 2
    is_efficient_backbone: bool = False
