"""Stable Diffusion pipeline (standard v1.5 and efficient versions) in Flax Linen (jax.nn)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import (
    AutoencoderKLConfig,
    CLIPTextConfig,
    StableDiffusionConfig,
    UNet2DConditionConfig,
)
from jax_model_zoo.models.clip.clip import CLIPTextTransformer
from jax_model_zoo.models.stable_diffusion.unet import UNet2DConditionModel
from jax_model_zoo.models.stable_diffusion.vae import AutoencoderKL


class StableDiffusionPipeline(nn.Module):
    """
    Unified Stable Diffusion model supporting standard v1.5 and Efficient (Tiny-SD) versions.
    Encapsulates VAE, UNet, and Text Encoder in Flax Linen.
    """

    config: StableDiffusionConfig

    def setup(self):
        self.vae = AutoencoderKL(config=self.config.vae)
        self.unet = UNet2DConditionModel(config=self.config.unet)
        self.text_encoder = CLIPTextTransformer(config=self.config.text_encoder)

    def encode_prompt(self, input_ids: jax.Array) -> jax.Array:
        """Encode prompt input IDs to text embeddings for UNet cross-attention."""
        _, hidden_states = self.text_encoder(input_ids, return_hidden_states=True)
        return hidden_states

    def encode_image(self, x: jax.Array, sample_posterior: bool = False, rng: Optional[jax.Array] = None) -> jax.Array:
        """Encode image to scaled latent representation."""
        posterior = self.vae.encode(x)
        latents = posterior.sample(rng) if sample_posterior else posterior.mode()
        return latents * self.config.vae.scaling_factor

    def decode_latents(self, latents: jax.Array) -> jax.Array:
        """Decode scaled latents back to image space."""
        z = latents / self.config.vae.scaling_factor
        return self.vae.decode(z)

    def predict_noise(
        self,
        latents: jax.Array,
        timestep: jax.Array,
        encoder_hidden_states: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Predict noise residual using UNet backbone."""
        return self.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        is_efficient: bool = False,
    ) -> Tuple[StableDiffusionPipeline, Dict[str, Any]]:
        """Load pretrained StableDiffusion pipeline components."""
        from jax_model_zoo.models.stable_diffusion.unet import UNet2DConditionModel
        from jax_model_zoo.models.stable_diffusion.vae import AutoencoderKL

        vae_model, vae_params = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        unet_model, unet_params = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        cfg = StableDiffusionConfig(
            vae=vae_model.config,
            unet=unet_model.config,
            is_efficient=is_efficient,
        )
        pipeline = cls(config=cfg)
        params = {
            "params": {
                "vae": vae_params.get("params", vae_params),
                "unet": unet_params.get("params", unet_params),
            }
        }
        return pipeline, params


def stable_diffusion_v1_5(**kwargs) -> StableDiffusionPipeline:
    """Standard Stable Diffusion v1.5 factory function."""
    cfg = StableDiffusionConfig(
        vae=AutoencoderKLConfig(block_out_channels=(128, 256, 512, 512), latent_channels=4),
        unet=UNet2DConditionConfig(block_out_channels=(320, 640, 1280, 1280), cross_attention_dim=768),
        is_efficient=False,
        **kwargs,
    )
    return StableDiffusionPipeline(config=cfg)


def stable_diffusion_efficient(**kwargs) -> StableDiffusionPipeline:
    """Efficient Stable Diffusion (Tiny-SD architecture: 3 stages 320/640/1280) factory function."""
    cfg = StableDiffusionConfig(
        vae=AutoencoderKLConfig(block_out_channels=(128, 256, 512, 512), latent_channels=4),
        unet=UNet2DConditionConfig(
            block_out_channels=(320, 640, 1280),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=768,
        ),
        is_efficient=True,
        **kwargs,
    )
    return StableDiffusionPipeline(config=cfg)
