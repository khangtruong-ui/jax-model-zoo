"""End-to-end numerical parity tests verifying that JAX implementations match
PyTorch reference models tensor-for-tensor and embedding-for-embedding.
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


@pytest.mark.timeout(60)
def test_vit_tensor_parity():
    """Verify VisionTransformer produces numerically identical embeddings to timm ViT."""
    import timm
    from jax_model_zoo.models.vision_transformer.vit import VisionTransformer, ViTConfig
    from jax_model_zoo.converters.torch_to_jax import convert_torch_vit_to_jax

    torch.manual_seed(42)
    torch_vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    torch_vit.eval()

    cfg = ViTConfig(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        num_classes=10,
    )

    x_torch = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_torch = torch_vit(x_torch).numpy()

    jax_vit = VisionTransformer(config=cfg)
    jax_params = convert_torch_vit_to_jax(torch_vit.state_dict(), cfg)

    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    y_jax = np.array(jax_vit.apply(jax_params, x_jax, deterministic=True))

    np.testing.assert_allclose(y_jax, y_torch, atol=1e-4, rtol=1e-4)


@pytest.mark.timeout(60)
def test_efficientnet_tensor_parity():
    """Verify EfficientNet-B2 produces numerically identical logits to timm EfficientNet-B2."""
    import timm
    from jax_model_zoo.models.efficientnet.efficientnet import EfficientNet, EfficientNetConfig
    from jax_model_zoo.converters.torch_to_jax import convert_torch_efficientnet_to_jax

    torch.manual_seed(42)
    torch_eff = timm.create_model(
        "efficientnet_b2",
        pretrained=False,
        num_classes=10,
    )
    torch_eff.eval()

    cfg = EfficientNetConfig(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        image_size=64,
        num_classes=10,
    )

    x_torch = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y_torch = torch_eff(x_torch).numpy()

    jax_eff = EfficientNet(config=cfg)
    jax_params = convert_torch_efficientnet_to_jax(torch_eff.state_dict(), cfg)

    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    y_jax = np.array(jax_eff.apply(jax_params, x_jax, deterministic=True))

    np.testing.assert_allclose(y_jax, y_torch, atol=5e-4, rtol=5e-4)


@pytest.mark.timeout(60)
def test_mae_encoder_tensor_parity():
    """Verify MAE backbone representations match timm MAE forward_features."""
    import timm
    from jax_model_zoo.models.vision_transformer.mae import MaskedAutoencoder, MAEConfig
    from jax_model_zoo.converters.torch_to_jax import convert_torch_mae_to_jax

    torch.manual_seed(42)
    torch_vit = timm.create_model(
        "vit_base_patch16_224.mae",
        pretrained=False,
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )
    torch_vit.eval()

    cfg = MAEConfig(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
    )

    x_torch = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_torch_feats = torch_vit.forward_features(x_torch).numpy()

    jax_mae = MaskedAutoencoder(config=cfg)
    jax_params = convert_torch_mae_to_jax(torch_vit.state_dict(), cfg)

    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    y_jax_feats = np.array(jax_mae.apply(jax_params, x_jax, method=jax_mae.extract_features))

    np.testing.assert_allclose(y_jax_feats, y_torch_feats, atol=2e-4, rtol=2e-4)


@pytest.mark.timeout(60)
def test_clip_image_and_text_tensor_parity():
    """Verify CLIP vision and text normalized feature embeddings match HuggingFace CLIPModel."""
    from transformers import CLIPModel, CLIPConfig as HFCLIPConfig
    from jax_model_zoo.models.clip.clip import CLIP, CLIPConfig, CLIPVisionConfig, CLIPTextConfig
    from jax_model_zoo.converters.torch_to_jax import convert_torch_clip_to_jax

    torch.manual_seed(42)
    hf_cfg = HFCLIPConfig(
        projection_dim=32,
        _attn_implementation="eager",
        vision_config={
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "_attn_implementation": "eager",
        },
        text_config={
            "vocab_size": 100,
            "max_position_embeddings": 16,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "_attn_implementation": "eager",
            "eos_token_id": 99,
        },
    )
    torch_model = CLIPModel(hf_cfg)
    torch_model.eval()

    pixel_values = torch.randn(2, 3, 16, 16)
    input_ids = torch.tensor([[1, 5, 12, 99], [2, 10, 20, 99]], dtype=torch.long)

    with torch.no_grad():
        torch_img_out = torch_model.get_image_features(pixel_values=pixel_values)
        torch_img = torch_img_out.pooler_output if hasattr(torch_img_out, "pooler_output") else torch_img_out
        torch_img_norm = (torch_img / torch_img.norm(dim=-1, keepdim=True)).numpy()

        torch_txt_out = torch_model.get_text_features(input_ids=input_ids)
        torch_txt = torch_txt_out.pooler_output if hasattr(torch_txt_out, "pooler_output") else torch_txt_out
        torch_txt_norm = (torch_txt / torch_txt.norm(dim=-1, keepdim=True)).numpy()

    cfg = CLIPConfig(
        vision=CLIPVisionConfig(image_size=16, patch_size=4, width=32, layers=2, heads=4, output_dim=32, mlp_dim=64),
        text=CLIPTextConfig(context_length=16, vocab_size=100, width=32, heads=4, layers=2, output_dim=32, mlp_dim=64),
    )
    jax_clip = CLIP(config=cfg)
    jax_params = convert_torch_clip_to_jax(torch_model.state_dict(), cfg)

    px_jax = jnp.array(pixel_values.permute(0, 2, 3, 1).numpy())
    id_jax = jnp.array(input_ids.numpy())

    jax_img = np.array(jax_clip.apply(jax_params, px_jax, method=jax_clip.encode_image))
    jax_txt = np.array(jax_clip.apply(jax_params, id_jax, method=jax_clip.encode_text))

    np.testing.assert_allclose(jax_img, torch_img_norm, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(jax_txt, torch_txt_norm, atol=5e-3, rtol=5e-3)


@pytest.mark.timeout(60)
def test_stable_diffusion_vae_tensor_parity():
    """Verify AutoencoderKL encodes and decodes latents matching diffusers AutoencoderKL."""
    from diffusers import AutoencoderKL as TorchVAE
    from jax_model_zoo.models.stable_diffusion.vae import AutoencoderKL as JaxVAE, AutoencoderKLConfig
    from jax_model_zoo.converters.torch_to_jax import convert_torch_vae_to_jax

    torch.manual_seed(42)
    torch_vae = TorchVAE(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64),
        latent_channels=4,
        norm_num_groups=16,
        layers_per_block=1,
    )
    torch_vae.eval()

    cfg = AutoencoderKLConfig(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64),
        latent_channels=4,
        norm_num_groups=16,
        layers_per_block=1,
    )
    jax_vae = JaxVAE(config=cfg)
    jax_params = convert_torch_vae_to_jax(torch_vae.state_dict(), cfg)

    x_torch = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        latents_torch = torch_vae.encode(x_torch).latent_dist.mode().numpy()
        recon_torch = torch_vae.decode(torch.from_numpy(latents_torch)).sample.numpy()

    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    posterior_jax = jax_vae.apply(jax_params, x_jax, method=jax_vae.encode)
    latents_jax_nhwc = np.array(posterior_jax.mode())
    latents_jax_nchw = np.transpose(latents_jax_nhwc, (0, 3, 1, 2))

    recon_jax_nhwc = np.array(jax_vae.apply(jax_params, jnp.array(latents_jax_nhwc), method=jax_vae.decode))
    recon_jax_nchw = np.transpose(recon_jax_nhwc, (0, 3, 1, 2))

    np.testing.assert_allclose(latents_jax_nchw, latents_torch, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(recon_jax_nchw, recon_torch, atol=3e-4, rtol=3e-4)


@pytest.mark.timeout(60)
def test_stable_diffusion_unet_tensor_parity():
    """Verify UNet2DConditionModel produces matching predictions to diffusers UNet2DConditionModel."""
    from diffusers import UNet2DConditionModel as TorchUNet
    from jax_model_zoo.models.stable_diffusion.unet import UNet2DConditionModel as JaxUNet, UNet2DConditionConfig
    from jax_model_zoo.converters.torch_to_jax import convert_torch_unet_to_jax

    torch.manual_seed(42)
    torch_unet = TorchUNet(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=16,
    )
    torch_unet.eval()

    cfg = UNet2DConditionConfig(
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        norm_num_groups=16,
        layers_per_block=1,
    )
    flax_unet = JaxUNet(config=cfg)

    x_torch = torch.randn(1, 4, 16, 16)
    t_torch = torch.tensor([50], dtype=torch.long)
    ctx_torch = torch.randn(1, 5, 32)

    with torch.no_grad():
        y_torch = torch_unet(x_torch, t_torch, encoder_hidden_states=ctx_torch).sample.numpy()

    jax_params = convert_torch_unet_to_jax(torch_unet.state_dict(), cfg)

    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    t_jax = jnp.array(t_torch.numpy())
    ctx_jax = jnp.array(ctx_torch.numpy())

    y_jax = flax_unet.apply(jax_params, x_jax, t_jax, encoder_hidden_states=ctx_jax)
    y_jax_nchw = np.transpose(np.array(y_jax), (0, 3, 1, 2))

    np.testing.assert_allclose(y_jax_nchw, y_torch, atol=2e-4, rtol=2e-4)


@pytest.mark.timeout(60)
def test_diffusion_diff_blocks_tensor_parity():
    """Verify DiffusionDiff building blocks match PyTorch reference implementation."""
    import torch.nn as nn
    import torch.nn.functional as F
    from jax_model_zoo.models.diffusion_diff.blocks import ResidualConvBlock, sinusoidal_embedding as jax_sinusoidal
    from jax_model_zoo.converters.torch_to_jax import conv2d_weight_torch_to_jax

    # 1. Sinusoidal embedding parity
    def torch_sinusoidal(timesteps, dim, max_period=10000.0):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        args = timesteps[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    t = torch.tensor([10.0, 50.0, 100.0, 999.0])
    emb_t = torch_sinusoidal(t, 64).numpy()
    emb_j = np.array(jax_sinusoidal(jnp.array(t.numpy()), 64))
    np.testing.assert_allclose(emb_j, emb_t, atol=5e-5, rtol=5e-5)

    # 2. ResidualConvBlock parity
    class TorchResidualConvBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
            self.norm1 = nn.BatchNorm2d(out_c, eps=1e-5)
            self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
            self.norm2 = nn.BatchNorm2d(out_c, eps=1e-5)

        def forward(self, x):
            res = x
            h = F.silu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            return F.silu(h + res)

    torch.manual_seed(42)
    t_block = TorchResidualConvBlock(16, 16)
    t_block.eval()

    x_t = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        y_t = t_block(x_t).permute(0, 2, 3, 1).numpy()

    w1 = conv2d_weight_torch_to_jax(t_block.conv1.weight.detach().numpy())
    w2 = conv2d_weight_torch_to_jax(t_block.conv2.weight.detach().numpy())

    params = {
        "params": {
            "conv1": {"kernel": w1},
            "norm1": {"scale": t_block.norm1.weight.detach().numpy(), "bias": t_block.norm1.bias.detach().numpy()},
            "conv2": {"kernel": w2},
            "norm2": {"scale": t_block.norm2.weight.detach().numpy(), "bias": t_block.norm2.bias.detach().numpy()},
        },
        "batch_stats": {
            "norm1": {"mean": t_block.norm1.running_mean.detach().numpy(), "var": t_block.norm1.running_var.detach().numpy()},
            "norm2": {"mean": t_block.norm2.running_mean.detach().numpy(), "var": t_block.norm2.running_var.detach().numpy()},
        },
    }

    j_block = ResidualConvBlock(in_channels=16, out_channels=16)
    x_j = jnp.array(x_t.permute(0, 2, 3, 1).numpy())
    y_j = np.array(j_block.apply(params, x_j, deterministic=True))

    np.testing.assert_allclose(y_j, y_t, atol=1e-5, rtol=1e-5)
