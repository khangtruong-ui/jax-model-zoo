"""Vision Transformer (ViT-B, ViT-L) implemented in Flax Linen (jax.nn)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import ViTConfig


class ViTPatchEmbedding(nn.Module):
    """Image to Patch Embedding using 2D Convolution."""

    patch_size: int = 16
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # x is [B, H, W, C] in Flax convention
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="projection",
        )(x)
        # Flatten spatial dimensions: [B, H_patches, W_patches, embed_dim] -> [B, N, embed_dim]
        b, h, w, c = x.shape
        return x.reshape((b, h * w, c))


class ViTAttention(nn.Module):
    """Multi-Head Self-Attention for Vision Transformer."""

    dim: int
    num_heads: int = 12
    qkv_bias: bool = True
    attention_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        b, n, c = x.shape
        head_dim = self.dim // self.num_heads

        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")(x)
        qkv = qkv.reshape((b, n, 3, self.num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = jnp.array(1.0 / np.sqrt(head_dim), dtype=q.dtype)
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        if not deterministic and self.attention_dropout > 0.0:
            attn_weights = nn.Dropout(rate=self.attention_dropout)(attn_weights, deterministic=deterministic)

        out = jnp.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        out = jnp.swapaxes(out, 1, 2).reshape((b, n, self.dim))
        out = nn.Dense(self.dim, use_bias=True, name="out_proj")(out)
        return out


class ViTMLP(nn.Module):
    """Transformer MLP block with GELU activation."""

    dim: int
    mlp_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        x = nn.Dense(self.mlp_dim, name="fc1")(x)
        x = jax.nn.gelu(x, approximate=False)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(self.dim, name="fc2")(x)
        if not deterministic and self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        return x


class ViTBlock(nn.Module):
    """Transformer Encoder Block."""

    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # Pre-LN Self-Attention
        norm1 = nn.LayerNorm(epsilon=self.layer_norm_eps, name="norm1")(x)
        attn = ViTAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attention_dropout=self.attention_dropout,
            name="attn",
        )(norm1, deterministic=deterministic)
        x = x + attn

        # Pre-LN MLP
        norm2 = nn.LayerNorm(epsilon=self.layer_norm_eps, name="norm2")(x)
        mlp = ViTMLP(
            dim=self.dim,
            mlp_dim=int(self.dim * self.mlp_ratio),
            dropout=self.dropout,
            name="mlp",
        )(norm2, deterministic=deterministic)
        x = x + mlp
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) in Flax Linen.
    Backbone supporting ViT-B, ViT-L, intermediate feature returns, and head surgery.
    """

    config: ViTConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = True,
        return_features: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, List[jax.Array]]]:
        b = x.shape[0]
        # 1. Patch embedding
        x = ViTPatchEmbedding(
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            name="patch_embed",
        )(x)

        # 2. Add CLS token
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.embed_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (b, 1, self.config.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # 3. Add Positional embedding
        num_patches = (self.config.img_size // self.config.patch_size) ** 2
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.config.embed_dim),
        )
        x = x + pos_embed

        if not deterministic and self.config.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)

        # 4. Encoder Blocks
        intermediate_features = []
        for i in range(self.config.depth):
            x = ViTBlock(
                dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                qkv_bias=self.config.qkv_bias,
                dropout=self.config.dropout_rate,
                attention_dropout=self.config.attention_dropout_rate,
                layer_norm_eps=self.config.layer_norm_eps,
                name=f"block_{i}",
            )(x, deterministic=deterministic)
            intermediate_features.append(x)

        # 5. Final LayerNorm
        x = nn.LayerNorm(epsilon=self.config.layer_norm_eps, name="norm")(x)

        # 6. Classification Head
        if self.config.classifier == "token":
            rep = x[:, 0]
        elif self.config.classifier == "gap":
            rep = jnp.mean(x[:, 1:], axis=1)
        else:
            rep = x

        logits = rep
        if self.config.num_classes > 0 and self.config.classifier != "none":
            logits = nn.Dense(self.config.num_classes, name="head")(rep)

        if return_features:
            return logits, intermediate_features
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "google/vit-base-patch16-224",
    ) -> Tuple[VisionTransformer, Dict[str, Any]]:
        """Load pretrained Vision Transformer weights directly from HuggingFace."""
        from transformers import ViTForImageClassification
        from jax_model_zoo.converters.torch_to_jax import convert_torch_vit_to_jax

        torch_model = ViTForImageClassification.from_pretrained(model_name)
        cfg = cls._hf_to_config(torch_model.config)
        flax_model = cls(config=cfg)

        jax_params = convert_torch_vit_to_jax(torch_model.state_dict(), cfg)
        return flax_model, jax_params

    @staticmethod
    def _hf_to_config(hf_cfg: Any) -> ViTConfig:
        return ViTConfig(
            img_size=getattr(hf_cfg, "image_size", 224),
            patch_size=getattr(hf_cfg, "patch_size", 16),
            embed_dim=getattr(hf_cfg, "hidden_size", 768),
            depth=getattr(hf_cfg, "num_hidden_layers", 12),
            num_heads=getattr(hf_cfg, "num_attention_heads", 12),
            mlp_ratio=getattr(hf_cfg, "intermediate_size", 3072) / getattr(hf_cfg, "hidden_size", 768),
            num_classes=getattr(hf_cfg, "num_labels", 1000),
            qkv_bias=getattr(hf_cfg, "qkv_bias", True),
            layer_norm_eps=getattr(hf_cfg, "layer_norm_eps", 1e-12),
        )


def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> Union[VisionTransformer, Tuple[VisionTransformer, Dict[str, Any]]]:
    """ViT-Base/16 factory function."""
    if pretrained:
        return VisionTransformer.from_pretrained("google/vit-base-patch16-224")
    cfg = ViTConfig(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )
    return VisionTransformer(config=cfg)


def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> Union[VisionTransformer, Tuple[VisionTransformer, Dict[str, Any]]]:
    """ViT-Large/16 factory function."""
    if pretrained:
        return VisionTransformer.from_pretrained("google/vit-large-patch16-224")
    cfg = ViTConfig(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )
    return VisionTransformer(config=cfg)


def vit_large_patch14_224(pretrained: bool = False, **kwargs) -> Union[VisionTransformer, Tuple[VisionTransformer, Dict[str, Any]]]:
    """ViT-Large/14 factory function."""
    cfg = ViTConfig(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )
    return VisionTransformer(config=cfg)
