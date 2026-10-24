"""Diffusion-Diff-V2 with Perpendicular Skip Connections in Flax Linen (jax.nn)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax_model_zoo.core.config import AutoencoderKLConfig, DiffusionDiffConfig, UNet2DConditionConfig
from jax_model_zoo.models.diffusion_diff.blocks import (
    AuxiliaryClassifier,
    DecoderUpsampleStage,
    PerpendicularSkipFusion,
    expand_to_spatial,
    get_valid_group_norm_groups,
    sinusoidal_embedding,
)
from jax_model_zoo.models.stable_diffusion.unet import UNet2DConditionModel
from jax_model_zoo.models.stable_diffusion.vae import AutoencoderKL


class TrainableLatentDecoderV2(nn.Module):
    """
    Trainable Latent Decoder V2 with Perpendicular Skip Connection Fusion.
    Fuses parallel frozen VAE decoder features directly into upsampling stages.
    """

    channels: List[int]
    out_channels: int = 1
    upsample_mode: str = "bilinear"
    norm_type: str = "batchnorm"
    dropout_rate: float = 0.0
    num_res_blocks: int = 1

    @nn.compact
    def __call__(
        self,
        z: jax.Array,
        perp_skips: Optional[List[jax.Array]] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        # 1. Initial projection
        x = nn.Conv(self.channels[0], kernel_size=(3, 3), padding="SAME", use_bias=False, name="init_conv_0")(z)
        if self.norm_type == "batchnorm":
            x = nn.BatchNorm(use_running_average=deterministic, name="init_conv_1")(x)
        elif self.norm_type == "groupnorm":
            x = nn.GroupNorm(num_groups=min(32, self.channels[0]), name="init_conv_1")(x)
        x = jax.nn.silu(x)

        # 2. Upsampling stages with Perpendicular Skip Fusions
        for i in range(1, len(self.channels)):
            next_ch = self.channels[i]
            x = DecoderUpsampleStage(
                in_channels=self.channels[i - 1],
                out_channels=next_ch,
                upsample_mode=self.upsample_mode,
                norm_type=self.norm_type,
                dropout_rate=self.dropout_rate,
                num_res_blocks=self.num_res_blocks,
                name=f"stages_{i - 1}",
            )(x, deterministic=deterministic)

            if perp_skips is not None and (i - 1) < len(perp_skips):
                perp = perp_skips[i - 1]
                if perp is not None:
                    x = PerpendicularSkipFusion(
                        dec_channels=next_ch,
                        perp_channels=perp.shape[-1],
                        norm_type=self.norm_type,
                        dropout_rate=self.dropout_rate,
                        name=f"perp_skip_fusions_{i - 1}",
                    )(x, perp, deterministic=deterministic)

        # 3. Final projection
        return nn.Conv(self.out_channels, kernel_size=(3, 3), padding="SAME", name="out_conv")(x)


class DiffusionDiffV2(nn.Module):
    """Diffusion-Diff-V2 model with parallel frozen decoder and perpendicular skip connections."""

    config: DiffusionDiffConfig

    def setup(self):
        # Noise schedule
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float32) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.asarray(np.cumprod(alphas, axis=0))

        # Frozen components
        vae_cfg = AutoencoderKLConfig(
            block_out_channels=(32, 32, 32, 64) if self.config.is_efficient_backbone else (128, 256, 512, 512),
            latent_channels=4,
        )
        self.vae = AutoencoderKL(config=vae_cfg)

        unet_cfg = UNet2DConditionConfig(
            block_out_channels=(32, 32, 32, 64) if self.config.is_efficient_backbone else (320, 640, 1280, 1280),
            cross_attention_dim=32 if self.config.is_efficient_backbone else 768,
        )
        self.diffuser = UNet2DConditionModel(config=unet_cfg)

        self.decoder = TrainableLatentDecoderV2(
            channels=self.config.decoder_channels,
            out_channels=self.config.out_channels,
            norm_type=self.config.decoder_norm,
            dropout_rate=self.config.decoder_dropout,
            num_res_blocks=self.config.decoder_res_blocks,
        )

        if self.config.aux_classifier:
            self.classifier_head = AuxiliaryClassifier(
                in_channels=self.total_z_channels,
                num_classes=self.config.num_classes,
                dropout_rate=self.config.decoder_dropout,
            )

    @property
    def total_z_channels(self) -> int:
        """Calculate total number of concatenated channels in representation Z."""
        c = 0
        latent_channels = 4
        if self.config.include_z0:
            c += latent_channels
        per_step = 0
        if self.config.include_noisy_latents:
            per_step += latent_channels
        if self.config.include_added_noise:
            per_step += latent_channels
        if self.config.include_predicted_noise:
            per_step += latent_channels
        if self.config.include_noise_diff:
            per_step += latent_channels
        per_step += self.config.timestep_embed_dim
        per_step += self.config.sigma_embed_dim

        c += len(self.config.timesteps) * per_step
        return c

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        rng: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        b, h, w, c = x.shape

        # 1. Clean latent z0 from frozen VAE encoder
        posterior = self.vae.encode(x)
        z0 = posterior.mode() * self.config.scaling_factor
        z0 = jnp.clip(z0, -15.0, 15.0)
        _, h_z, w_z, _ = z0.shape

        # 2. Parallel frozen VAE decoder produces perpendicular skip features
        z0_for_decoder = z0 / self.config.scaling_factor
        _, parallel_decoder_features = self.vae.decode(z0_for_decoder, return_features=True)
        # Select matching stages for perpendicular fusion
        perp_skips = parallel_decoder_features[: len(self.config.decoder_channels) - 1]

        z_components: List[jax.Array] = []
        if self.config.include_z0:
            z_components.append(z0)

        # 3. Noise perturbation & diffuser predictions
        curr_rng = rng if rng is not None else jax.random.PRNGKey(42)

        for t_val in self.config.timesteps:
            t_clamped = max(0, min(t_val, len(self.alphas_cumprod) - 1))
            alpha_bar = self.alphas_cumprod[t_clamped]
            sigma_val = jnp.sqrt(jnp.maximum(1.0 - alpha_bar, 1e-8))
            sqrt_alpha_bar = jnp.sqrt(alpha_bar)

            curr_rng, subkey = jax.random.split(curr_rng)
            eps_k = jax.random.normal(subkey, z0.shape, dtype=z0.dtype)
            z_tk = sqrt_alpha_bar * z0 + sigma_val * eps_k

            t_tensor = jnp.full((b,), t_clamped, dtype=jnp.int32)
            cross_dim = self.diffuser.config.cross_attention_dim
            uncond_emb = jnp.zeros((b, 1, cross_dim), dtype=z_tk.dtype)

            pred_eps = self.diffuser(z_tk, t_tensor, encoder_hidden_states=uncond_emb)

            t_emb = sinusoidal_embedding(t_tensor.astype(jnp.float32), dim=self.config.timestep_embed_dim)
            t_spatial = expand_to_spatial(t_emb, h_z, w_z)

            sigma_tensor = jnp.broadcast_to(sigma_val[None], (b,))
            sigma_emb = sinusoidal_embedding(sigma_tensor, dim=self.config.sigma_embed_dim)
            sigma_spatial = expand_to_spatial(sigma_emb, h_z, w_z)

            if self.config.include_noisy_latents:
                z_components.append(z_tk)
            if self.config.include_added_noise:
                z_components.append(eps_k)
            if self.config.include_predicted_noise:
                z_components.append(pred_eps)
            if self.config.include_noise_diff:
                z_components.append(pred_eps - eps_k)
            if self.config.timestep_embed_dim > 0:
                z_components.append(t_spatial)
            if self.config.sigma_embed_dim > 0:
                z_components.append(sigma_spatial)

        # 4. Form Z representation
        z_high_dim = jnp.concatenate(z_components, axis=-1)
        z_groups = get_valid_group_norm_groups(z_high_dim.shape[-1])
        z_high_dim = nn.GroupNorm(num_groups=z_groups, name="z_norm")(z_high_dim)

        class_logits = None
        if self.config.aux_classifier:
            class_logits = self.classifier_head(z_high_dim, deterministic=deterministic)

        mask_logits = self.decoder(z_high_dim, perp_skips=perp_skips, deterministic=deterministic)

        # Match input resolution (H, W)
        if mask_logits.shape[1:3] != (h, w):
            mask_logits = jax.image.resize(mask_logits, (b, h, w, mask_logits.shape[-1]), method="bilinear")

        if self.config.aux_classifier:
            return mask_logits, class_logits
        return mask_logits


# Aliases matching ai-detection-unet-gpu
DiffusionDiffV2Model = DiffusionDiffV2
