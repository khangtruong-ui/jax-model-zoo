"""End-to-end numerical parity tests verifying that JAX implementations loaded
directly from online pretrained reputable repositories match PyTorch reference models.
"""

import os
import numpy as np
import pytest
import torch
import jax.numpy as jnp

from jax_model_zoo.models.vision_transformer.vit import vit_base_patch16_224, VisionTransformer
from jax_model_zoo.models.vision_transformer.mae import mae_vit_base_patch16, MaskedAutoencoder
from jax_model_zoo.models.efficientnet.efficientnet import efficientnet_b2, EfficientNet
from jax_model_zoo.models.clip.clip import clip_vit_base_patch32, CLIP
from jax_model_zoo.core.registry import download_and_convert, list_models, get_model_info


@pytest.mark.timeout(120)
def test_vit_base_online_pretrained_parity():
    """Verify ViT-Base with real HuggingFace pretrained weights matches PyTorch reference output."""
    from transformers import ViTForImageClassification

    torch.manual_seed(42)
    x_torch = torch.randn(1, 3, 224, 224)

    # 1. Reference PyTorch model
    torch_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").eval()
    with torch.no_grad():
        y_torch = torch_model(x_torch).logits.numpy()

    # 2. JAX model loaded via pretrained=True online loader
    jax_model, jax_params = vit_base_patch16_224(pretrained=True)
    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    y_jax = np.array(jax_model.apply(jax_params, x_jax, deterministic=True))

    # Check top-5 classification prediction alignment
    top5_torch = np.argsort(y_torch[0])[-5:]
    top5_jax = np.argsort(y_jax[0])[-5:]
    assert np.array_equal(top5_torch, top5_jax), (
        f"Top-5 class mismatch: torch={top5_torch}, jax={top5_jax}"
    )

    # Check numerical closeness
    max_diff = np.max(np.abs(y_torch - y_jax))
    assert max_diff < 0.05, f"Logit difference too high: {max_diff}"


@pytest.mark.timeout(120)
def test_efficientnet_b2_online_pretrained_parity():
    """Verify EfficientNet-B2 with real timm/ImageNet pretrained weights matches PyTorch reference output."""
    import timm

    torch.manual_seed(42)
    x_torch = torch.randn(1, 3, 288, 288)

    # 1. Reference PyTorch model
    torch_model = timm.create_model("efficientnet_b2", pretrained=True).eval()
    with torch.no_grad():
        y_torch = torch_model(x_torch).numpy()

    # 2. JAX model loaded via pretrained=True online loader
    jax_model, jax_params = efficientnet_b2(pretrained=True)
    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    y_jax = np.array(jax_model.apply(jax_params, x_jax, deterministic=True))

    # Numerical difference assertion (< 1e-4)
    max_diff = np.max(np.abs(y_torch - y_jax))
    assert max_diff < 1e-4, f"EfficientNet-B2 output difference too high: {max_diff}"


@pytest.mark.timeout(120)
def test_clip_online_pretrained_parity():
    """Verify CLIP with real OpenAI/HuggingFace weights matches image and text embeddings."""
    from transformers import CLIPModel, CLIPTokenizer

    torch.manual_seed(42)
    model_id = "openai/clip-vit-base-patch32"

    # Reference PyTorch model
    torch_model = CLIPModel.from_pretrained(model_id).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_id)

    # JAX model
    jax_model, jax_params = clip_vit_base_patch32(pretrained=True)

    # 1. Image feature parity
    px_torch = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        img_torch = torch_model.get_image_features(pixel_values=px_torch).pooler_output
        img_torch = (img_torch / img_torch.norm(dim=-1, keepdim=True)).numpy()

    px_jax = jnp.array(px_torch.permute(0, 2, 3, 1).numpy())
    img_jax = np.array(jax_model.apply(jax_params, px_jax, method=jax_model.encode_image))
    img_diff = np.max(np.abs(img_torch - img_jax))
    assert img_diff < 1e-4, f"CLIP image embedding diff too high: {img_diff}"

    # 2. Text feature parity
    inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    with torch.no_grad():
        txt_torch = torch_model.get_text_features(**inputs).pooler_output
        txt_torch = (txt_torch / txt_torch.norm(dim=-1, keepdim=True)).numpy()

    id_jax = jnp.array(inputs["input_ids"].numpy())
    txt_jax = np.array(jax_model.apply(jax_params, id_jax, method=jax_model.encode_text))
    txt_diff = np.max(np.abs(txt_torch - txt_jax))
    assert txt_diff < 1e-4, f"CLIP text embedding diff too high: {txt_diff}"


@pytest.mark.timeout(120)
def test_mae_online_pretrained_parity():
    """Verify MAE unmasked encoder representations with real facebook/vit-mae-base weights."""
    from transformers import ViTMAEForPreTraining

    torch.manual_seed(42)
    x_torch = torch.randn(1, 3, 224, 224)

    # Reference PyTorch unmasked features
    torch_mae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").eval()
    with torch.no_grad():
        embs = torch_mae.vit.embeddings.patch_embeddings(x_torch)
        embs = embs + torch_mae.vit.embeddings.position_embeddings[:, 1:, :]
        cls_toks = (torch_mae.vit.embeddings.cls_token + torch_mae.vit.embeddings.position_embeddings[:, :1, :]).expand(embs.shape[0], -1, -1)
        embs = torch.cat((cls_toks, embs), dim=1)
        for layer in torch_mae.vit.layers:
            embs = layer(embs)
            if isinstance(embs, tuple):
                embs = embs[0]
        y_torch_feats = torch_mae.vit.layernorm(embs).numpy()

    # JAX model
    jax_mae, jax_params = mae_vit_base_patch16(pretrained=True)
    x_jax = jnp.array(x_torch.permute(0, 2, 3, 1).numpy())
    y_jax_feats = np.array(jax_mae.apply(jax_params, x_jax, method=jax_mae.extract_features))

    assert y_torch_feats.shape == y_jax_feats.shape
    diff = np.max(np.abs(y_torch_feats - y_jax_feats))
    assert diff < 0.05, f"MAE feature diff too high: {diff}"


@pytest.mark.timeout(120)
def test_registry_and_download_script(tmp_path):
    """Verify model registry catalog and download_and_convert utility."""
    models = list_models()
    assert "vit-b" in models
    assert "efficientnet-b2" in models
    assert "clip-vit-base-patch32" in models

    info = get_model_info("vit-b")
    assert info["online_source"] == "google/vit-base-patch16-224"
    assert info["source_type"] == "huggingface"

    # Test download and conversion into temporary directory
    model, params, dest = download_and_convert(
        "vit-b",
        output_dir=tmp_path,
        format="safetensors",
        verify=True,
    )
    assert dest.exists()
    assert dest.suffix == ".safetensors"
