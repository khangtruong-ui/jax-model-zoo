"""CLIP (Contrastive Language-Image Pre-Training) implemented in Flax Linen (jax.nn)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


def quick_gelu(x: jax.Array) -> jax.Array:
    """QuickGELU activation function used in OpenAI CLIP."""
    return x * jax.nn.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):
    d_model: int
    n_head: int

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        b, seq_len, _ = x.shape
        head_dim = self.d_model // self.n_head
        scale = 1.0 / np.sqrt(head_dim)

        q = nn.Dense(self.d_model, name="q_proj")(x)
        k = nn.Dense(self.d_model, name="k_proj")(x)
        v = nn.Dense(self.d_model, name="v_proj")(x)

        q = q.reshape((b, seq_len, self.n_head, head_dim)).swapaxes(1, 2)
        k = k.reshape((b, seq_len, self.n_head, head_dim)).swapaxes(1, 2)
        v = v.reshape((b, seq_len, self.n_head, head_dim)).swapaxes(1, 2)

        scores = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(weights, v).swapaxes(1, 2).reshape((b, seq_len, self.d_model))
        return nn.Dense(self.d_model, name="out_proj")(out)


class CLIPMLP(nn.Module):
    d_model: int
    mlp_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = self.mlp_dim or (self.d_model * 4)
        h = nn.Dense(dim, name="c_fc")(x)
        h = quick_gelu(h)
        return nn.Dense(self.d_model, name="c_proj")(h)


class ResidualAttentionBlock(nn.Module):
    d_model: int
    n_head: int
    mlp_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        norm1 = nn.LayerNorm(epsilon=1e-5, name="ln_1")(x)
        x = x + CLIPAttention(d_model=self.d_model, n_head=self.n_head, name="attn")(norm1, mask=mask)
        norm2 = nn.LayerNorm(epsilon=1e-5, name="ln_2")(x)
        x = x + CLIPMLP(d_model=self.d_model, mlp_dim=self.mlp_dim, name="mlp")(norm2)
        return x


class CLIPVisionTransformer(nn.Module):
    """Vision Transformer backbone for CLIP."""

    config: CLIPVisionConfig

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # x is [B, H, W, C]
        b = x.shape[0]

        # 1. Patch Conv
        x = nn.Conv(
            features=self.config.width,
            kernel_size=(self.config.patch_size, self.config.patch_size),
            strides=(self.config.patch_size, self.config.patch_size),
            use_bias=False,
            name="conv1",
        )(x)
        b, h, w, c = x.shape
        x = x.reshape((b, h * w, c))

        # 2. Add CLS token
        cls_token = self.param(
            "class_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.config.width,),
        )
        cls_tokens = jnp.broadcast_to(cls_token[None, None, :], (b, 1, self.config.width))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # 3. Add Positional embedding
        pos_embed = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=0.02),
            ((self.config.image_size // self.config.patch_size) ** 2 + 1, self.config.width),
        )
        x = x + pos_embed[None, :, :]

        # 4. Pre-LN
        x = nn.LayerNorm(epsilon=1e-5, name="ln_pre")(x)

        # 5. Transformer blocks
        for i in range(self.config.layers):
            x = ResidualAttentionBlock(
                d_model=self.config.width,
                n_head=self.config.heads,
                mlp_dim=self.config.mlp_dim,
                name=f"resblocks_{i}",
            )(x)

        # 6. Post-LN on CLS token
        x = nn.LayerNorm(epsilon=1e-5, name="ln_post")(x[:, 0])

        # 7. Projection head
        out = nn.Dense(self.config.output_dim, use_bias=False, name="proj")(x)
        return out


class CLIPTextTransformer(nn.Module):
    """Text Transformer backbone for CLIP."""

    config: CLIPTextConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        return_hidden_states: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        # input_ids: [B, context_length]
        b, seq_len = input_ids.shape

        # 1. Token & Positional embeddings
        token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.width,
            name="token_embedding",
        )(input_ids)

        pos_embed = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.config.context_length, self.config.width),
        )
        x = token_embed + pos_embed[:seq_len][None, :, :]

        # Causal mask for autoregressive/prefix attention
        mask = jnp.triu(jnp.full((seq_len, seq_len), -1e9), k=1)

        # 2. Transformer blocks
        hidden_states = []
        for i in range(self.config.layers):
            x = ResidualAttentionBlock(
                d_model=self.config.width,
                n_head=self.config.heads,
                mlp_dim=self.config.mlp_dim,
                name=f"resblocks_{i}",
            )(x, mask=mask)
            hidden_states.append(x)

        # 3. Final LayerNorm
        x_norm = nn.LayerNorm(epsilon=1e-5, name="ln_final")(x)

        # 4. Feature at EOT token
        eot_indices = jnp.argmax(input_ids, axis=-1)
        pooled = jnp.take_along_axis(x_norm, eot_indices[:, None, None], axis=1)[:, 0, :]

        # 5. Projection
        out = nn.Dense(self.config.output_dim, use_bias=False, name="text_projection")(pooled)

        if return_hidden_states:
            return out, x_norm
        return out

        if return_hidden_states:
            return out, x_norm
        return out


class CLIP(nn.Module):
    """Multimodal Contrastive Language-Image Pre-Training Model in Flax Linen."""

    config: CLIPConfig

    def setup(self):
        self.visual = CLIPVisionTransformer(config=self.config.vision)
        self.text = CLIPTextTransformer(config=self.config.text)
        self.logit_scale = self.param(
            "logit_scale",
            nn.initializers.constant(self.config.logit_scale_init),
            (),
        )

    def __call__(
        self,
        pixel_values: jax.Array,
        input_ids: jax.Array,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute logits per image and logits per text."""
        image_embeds = self.visual(pixel_values, deterministic=deterministic)
        text_embeds = self.text(input_ids)

        # Normalize features
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        scale = jnp.exp(self.logit_scale)

        logits_per_image = scale * jnp.matmul(image_embeds, text_embeds.T)
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    def encode_image(self, pixel_values: jax.Array, deterministic: bool = True) -> jax.Array:
        """Extract normalized image features."""
        feat = self.visual(pixel_values, deterministic=deterministic)
        return feat / jnp.linalg.norm(feat, axis=-1, keepdims=True)

    def encode_text(self, input_ids: jax.Array) -> jax.Array:
        """Extract normalized text features."""
        feat = self.text(input_ids)
        return feat / jnp.linalg.norm(feat, axis=-1, keepdims=True)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> Tuple[CLIP, Dict[str, Any]]:
        """Load pretrained CLIP model and weights from HuggingFace."""
        from transformers import CLIPModel
        from jax_model_zoo.converters.torch_to_jax import convert_torch_clip_to_jax

        torch_model = CLIPModel.from_pretrained(model_name)
        cfg = CLIPConfig(
            vision=CLIPVisionConfig(
                image_size=torch_model.config.vision_config.image_size,
                patch_size=torch_model.config.vision_config.patch_size,
                width=torch_model.config.vision_config.hidden_size,
                layers=torch_model.config.vision_config.num_hidden_layers,
                heads=torch_model.config.vision_config.num_attention_heads,
                output_dim=torch_model.config.projection_dim,
            ),
            text=CLIPTextConfig(
                context_length=torch_model.config.text_config.max_position_embeddings,
                vocab_size=torch_model.config.text_config.vocab_size,
                width=torch_model.config.text_config.hidden_size,
                heads=torch_model.config.text_config.num_attention_heads,
                layers=torch_model.config.text_config.num_hidden_layers,
                output_dim=torch_model.config.projection_dim,
            ),
        )
        flax_model = cls(config=cfg)
        jax_params = convert_torch_clip_to_jax(torch_model.state_dict(), cfg)
        return flax_model, jax_params


def clip_vit_base_patch32(pretrained: bool = False, **kwargs) -> Union[CLIP, Tuple[CLIP, Dict[str, Any]]]:
    """CLIP ViT-B/32 factory function."""
    if pretrained:
        return CLIP.from_pretrained("openai/clip-vit-base-patch32")
    cfg = CLIPConfig(
        vision=CLIPVisionConfig(patch_size=32, width=768, layers=12, heads=12, output_dim=512),
        text=CLIPTextConfig(width=512, heads=8, layers=12, output_dim=512),
        **kwargs,
    )
    return CLIP(config=cfg)


def clip_vit_base_patch16(pretrained: bool = False, **kwargs) -> Union[CLIP, Tuple[CLIP, Dict[str, Any]]]:
    """CLIP ViT-B/16 factory function."""
    if pretrained:
        return CLIP.from_pretrained("openai/clip-vit-base-patch16")
    cfg = CLIPConfig(
        vision=CLIPVisionConfig(patch_size=16, width=768, layers=12, heads=12, output_dim=512),
        text=CLIPTextConfig(width=512, heads=8, layers=12, output_dim=512),
        **kwargs,
    )
    return CLIP(config=cfg)


def clip_vit_large_patch14(pretrained: bool = False, **kwargs) -> Union[CLIP, Tuple[CLIP, Dict[str, Any]]]:
    """CLIP ViT-L/14 factory function."""
    if pretrained:
        return CLIP.from_pretrained("openai/clip-vit-large-patch14")
    cfg = CLIPConfig(
        vision=CLIPVisionConfig(patch_size=14, width=1024, layers=24, heads=16, output_dim=768),
        text=CLIPTextConfig(width=768, heads=12, layers=12, output_dim=768),
        **kwargs,
    )
    return CLIP(config=cfg)
