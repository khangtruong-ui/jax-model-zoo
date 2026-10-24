"""Masked Autoencoder (MAE) implemented in Flax Linen (jax.nn)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import MAEConfig
from jax_model_zoo.models.vision_transformer.vit import ViTBlock, ViTPatchEmbedding


def random_masking(
    x: jax.Array,
    mask_ratio: float,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Perform per-sample random masking of patches.
    x: [B, N, D], sequence of patch tokens
    Returns:
        x_masked: [B, len_keep, D]
        mask: [B, N], 0 is kept, 1 is removed
        ids_restore: [B, N], indices to restore original sequence
    """
    b, n, d = x.shape
    len_keep = int(n * (1.0 - mask_ratio))

    # Noise for sorting
    noise = jax.random.uniform(rng, (b, n))

    # Ascend: small is keep, large is remove
    ids_shuffle = jnp.argsort(noise, axis=1)
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # Keep the first len_keep patches
    ids_keep = ids_shuffle[:, :len_keep]
    # Gather kept patches: [B, len_keep, D]
    x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], axis=1)

    # Generate binary mask: 0 is keep, 1 is remove
    mask = jnp.ones((b, n), dtype=x.dtype)
    mask = mask.at[:, :len_keep].set(0.0)
    # Unshuffle to original order
    mask = jnp.take_along_axis(mask, ids_restore, axis=1)

    return x_masked, mask, ids_restore


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) with ViT backbone.
    Supports both self-supervised reconstruction and downstream feature extraction.
    """

    config: MAEConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask_ratio: Optional[float] = None,
        rng: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Reconstruction forward pass.
        Returns:
            pred: [B, N, patch_size * patch_size * in_channels]
            mask: [B, N] binary mask (1 = masked)
        """
        b = x.shape[0]
        ratio = self.config.mask_ratio if mask_ratio is None else mask_ratio

        # 1. Patch embedding
        patches = ViTPatchEmbedding(
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            name="patch_embed",
        )(x)

        # 2. Add position embedding (without cls token)
        num_patches = (self.config.img_size // self.config.patch_size) ** 2
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.config.embed_dim),
        )
        patches = patches + pos_embed

        # 3. Masking
        if ratio > 0.0:
            if rng is None:
                rng = jax.random.PRNGKey(0)
            x_masked, mask, ids_restore = random_masking(patches, ratio, rng)
        else:
            x_masked = patches
            mask = jnp.zeros((b, num_patches), dtype=patches.dtype)
            ids_restore = jnp.tile(jnp.arange(num_patches), (b, 1))

        # 4. Prepend CLS token
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.embed_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (b, 1, self.config.embed_dim))
        latent = jnp.concatenate([cls_tokens, x_masked], axis=1)

        # 5. Encoder Transformer Blocks
        for i in range(self.config.depth):
            latent = ViTBlock(
                dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                name=f"encoder_block_{i}",
            )(latent, deterministic=deterministic)

        latent = nn.LayerNorm(name="encoder_norm")(latent)

        # 6. Decoder projection
        dec_latent = nn.Dense(self.config.decoder_embed_dim, name="decoder_embed")(latent)

        # 7. Append mask tokens
        mask_token = self.param(
            "mask_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.decoder_embed_dim),
        )
        mask_tokens = jnp.broadcast_to(
            mask_token,
            (b, ids_restore.shape[1] + 1 - dec_latent.shape[1], self.config.decoder_embed_dim),
        )
        x_dec = jnp.concatenate([dec_latent[:, 1:], mask_tokens], axis=1)
        # Unshuffle to original patch sequence
        x_dec = jnp.take_along_axis(x_dec, ids_restore[:, :, None], axis=1)
        # Prepend decoder CLS token
        x_dec = jnp.concatenate([dec_latent[:, :1], x_dec], axis=1)

        # 8. Add Decoder Positional Embeddings
        decoder_pos_embed = self.param(
            "decoder_pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.config.decoder_embed_dim),
        )
        x_dec = x_dec + decoder_pos_embed

        # 9. Decoder Transformer Blocks
        for i in range(self.config.decoder_depth):
            x_dec = ViTBlock(
                dim=self.config.decoder_embed_dim,
                num_heads=self.config.decoder_num_heads,
                mlp_ratio=self.config.decoder_mlp_ratio,
                name=f"decoder_block_{i}",
            )(x_dec, deterministic=deterministic)

        x_dec = nn.LayerNorm(name="decoder_norm")(x_dec)

        # 10. Predict patch pixels
        patch_dim = (self.config.patch_size ** 2) * self.config.in_channels
        pred = nn.Dense(patch_dim, name="decoder_pred")(x_dec[:, 1:])

        return pred, mask

    @nn.compact
    def extract_features(
        self,
        x: jax.Array,
        deterministic: bool = True,
    ) -> jax.Array:
        """
        Extract backbone encoder representations without patch masking.
        Returns: [B, N + 1, embed_dim]
        """
        b = x.shape[0]
        patches = ViTPatchEmbedding(
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            name="patch_embed",
        )(x)

        num_patches = (self.config.img_size // self.config.patch_size) ** 2
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.config.embed_dim),
        )
        patches = patches + pos_embed

        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.config.embed_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (b, 1, self.config.embed_dim))
        latent = jnp.concatenate([cls_tokens, patches], axis=1)

        for i in range(self.config.depth):
            latent = ViTBlock(
                dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                name=f"encoder_block_{i}",
            )(latent, deterministic=deterministic)

        return nn.LayerNorm(name="encoder_norm")(latent)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "facebook/mae-vit-base",
    ) -> Tuple[MaskedAutoencoder, Dict[str, Any]]:
        """Load pretrained MAE weights directly from HuggingFace."""
        from transformers import ViTMAEForPreTraining
        from jax_model_zoo.converters.torch_to_jax import convert_torch_mae_to_jax

        torch_model = ViTMAEForPreTraining.from_pretrained(model_name)
        cfg = MAEConfig(
            img_size=torch_model.config.image_size,
            patch_size=torch_model.config.patch_size,
            embed_dim=torch_model.config.hidden_size,
            depth=torch_model.config.num_hidden_layers,
            num_heads=torch_model.config.num_attention_heads,
            decoder_embed_dim=torch_model.config.decoder_hidden_size,
            decoder_depth=torch_model.config.decoder_num_hidden_layers,
            decoder_num_heads=torch_model.config.decoder_num_attention_heads,
        )
        flax_model = cls(config=cfg)
        jax_params = convert_torch_mae_to_jax(torch_model.state_dict(), cfg)
        return flax_model, jax_params


def mae_vit_base_patch16(**kwargs) -> MaskedAutoencoder:
    """MAE ViT-Base factory."""
    cfg = MAEConfig(embed_dim=768, depth=12, num_heads=12, **kwargs)
    return MaskedAutoencoder(config=cfg)


def mae_vit_large_patch16(**kwargs) -> MaskedAutoencoder:
    """MAE ViT-Large factory."""
    cfg = MAEConfig(embed_dim=1024, depth=24, num_heads=16, **kwargs)
    return MaskedAutoencoder(config=cfg)
