# JAX Model Zoo (Flax Linen / `jax.nn`)

A high-performance, pip-installable JAX model library containing modern computer vision and generative deep learning backbones implemented in **Flax Linen (`jax.nn`)** (strictly avoiding `nnx`).

`jax-model-zoo` provides topology-invariant checkpointing (safeguarding workflows against Orbax topology mismatches across single/multi-GPU and TPU cores), bidirectional PyTorch $\leftrightarrow$ JAX weight conversions, model surgery utilities (LoRA injection/merging, head replacement, trainable parameter masks), FLOPs compute cost profilers, and single-host multi-core GPU/TPU scaling helpers.

---

## Key Features

1. **Pure Flax Linen (`jax.nn`) Architectures**:
   - Explicitly built on `flax.linen` modules with functional purity and full compatibility with `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap`.
   - **No randomly initialized placeholders**: Full support for loading pretrained weights or converting them directly from PyTorch / HuggingFace / timm.
2. **Integrated Backbones & Generative Architectures**:
   - **EfficientNet-B2**: Full inverted bottleneck blocks (MBConv), Squeeze-and-Excitation, and multi-scale feature pyramid extraction.
   - **Vision Transformer (ViT-B / ViT-L)**: Standard patch projections, multi-head self-attention, and intermediate representation extraction.
   - **Masked Autoencoder (MAE)**: Pretext self-supervised autoencoder, patch masking, asymmetric encoder-decoder, and representation extraction.
   - **CLIP (Vision + Text Dual Encoder)**: Joint multi-modal representation model with contrastive normalization and projection heads.
   - **Stable Diffusion (v1.5 & Efficient / Tiny-SD)**: Complete AutoencoderKL (VAE), UNet2DConditionModel (with cross-attention), text embeddings, and inference pipelines.
   - **DiffusionDiff Family**: Architecture and checkpoint signatures strictly reproduced from [`ai-detection-unet-gpu`](https://github.com/khangtruong-ui/ai-detection-unet-gpu):
     - `DiffusionDiff` (V1 baseline)
     - `DiffusionDiffV2` (Parallel decoder perpendicular skip fusion)
     - `DiffusionDiffMinimized` (2 lines of computation, perpendicular skip fusion, 84-channel Z representations)
3. **Device-Agnostic & Topology-Invariant Checkpoints**:
   - Checkpoints saved in Safetensors, Msgpack, or CPU-leaf Orbax formats that load reliably across changing device topologies (1 GPU, 8 GPUs, TPU v2/v3/v4 cores, or CPU) without Orbax mesh mismatch errors.
4. **Bidirectional Checkpoint Conversion (PyTorch $\leftrightarrow$ JAX)**:
   - Automated layout handling: Conv2D (`[O, I, H, W]` $\leftrightarrow$ `[H, W, I, O]`), Dense (`[O, I]` $\leftrightarrow$ `[I, O]`), and nested dictionary flattening/unflattening.
5. **Model Surgery**:
   - Low-Rank Adaptation (`LoRADense`, `LoRAConv`, `inject_lora`, `merge_lora`).
   - Task / classification head replacement (`replace_head`).
   - Optax parameter freezing boolean masks (`get_trainable_param_mask`).
6. **Compute Cost & FLOPs Export**:
   - Accurate Jaxpr operation analysis estimating theoretical FLOPs, GMACs, and parameter counts.
7. **Single-Host Multi-Device GPU / TPU Support**:
   - Data parallelism utilities (`replicate`, `unreplicate`, `shard_batch`, `create_data_parallel_apply`).

---

## Installation

Install in editable mode from source:
```bash
git clone https://github.com/khangtruong-ui/jax-model-zoo.git
cd jax-model-zoo
pip install -e .
```

To include all optional generative and weight conversion dependencies (e.g. `diffusers`, `torch`, `torchvision`, `safetensors`, `timm`):
```bash
pip install -e .[all]
```

---

## Quickstart

### 1. Vision Transformer (ViT-B & ViT-L)

```python
import jax
import jax.numpy as jnp
from jax_model_zoo import VisionTransformer, vit_base_patch16_224, vit_large_patch14_224

# Instantiate pretrained ViT-Base backbone
model, params = vit_base_patch16_224(pretrained=True, num_classes=1000)

x = jnp.ones((1, 224, 224, 3))
logits = model.apply(params, x)
print("ViT Logits Shape:", logits.shape)  # [1, 1000]

# Intermediate feature representation extraction
logits, hidden_states = model.apply(params, x, return_features=True)
print("Number of intermediate layers:", len(hidden_states))
```

### 2. Masked Autoencoder (MAE)

```python
from jax_model_zoo import MaskedAutoencoder, mae_vit_base_patch16

# Create MAE ViT-Base with 75% patch masking
model, params = mae_vit_base_patch16(pretrained=True)

rng = jax.random.PRNGKey(0)
images = jnp.ones((2, 224, 224, 3))

# Reconstruction pretext task
pred_pixels, mask = model.apply(params, images, rng=rng)
print("Reconstructed Patches:", pred_pixels.shape)  # [2, 196, 768]
print("Binary Mask (1=masked, 0=kept):", mask.shape)  # [2, 196]

# Extract representations for downstream transfer
features = model.apply(params, images, method=model.extract_features)
print("Encoder Features:", features.shape)  # [2, 197, 768]
```

### 3. EfficientNet-B2

```python
from jax_model_zoo import EfficientNet, efficientnet_b2

model, params = efficientnet_b2(pretrained=True, num_classes=1000)

x = jnp.ones((1, 288, 288, 3))
logits, stage_features = model.apply(params, x, return_features=True)

print("EfficientNet-B2 Logits:", logits.shape)  # [1, 1000]
for idx, feat in enumerate(stage_features):
    print(f"Stage {idx} features:", feat.shape)
```

### 4. CLIP (Dual Encoder)

```python
from jax_model_zoo import CLIP, clip_vit_base_patch32

clip_model, clip_params = clip_vit_base_patch32(pretrained=True)

images = jnp.ones((1, 224, 224, 3))
tokens = jnp.zeros((1, 77), dtype=jnp.int32)

# Compute similarity logits
logits_per_image, logits_per_text = clip_model.apply(clip_params, images, tokens)

# Extract normalized projection embeddings
image_embeds = clip_model.apply(clip_params, images, method=clip_model.encode_image)
text_embeds = clip_model.apply(clip_params, tokens, method=clip_model.encode_text)
```

### 5. Stable Diffusion (Standard v1.5 & Efficient / Tiny-SD)

```python
from jax_model_zoo import stable_diffusion_v1_5, stable_diffusion_efficient

# Standard v1.5 Pipeline components
pipeline_v15 = stable_diffusion_v1_5()
vae_params, unet_params, text_params = pipeline_v15.load_pretrained_weights()

# Fast Efficient Pipeline (Segmind Tiny-SD with reduced stages)
pipeline_eff = stable_diffusion_efficient()
print("Efficient UNet stages:", len(pipeline_eff.config.unet.block_out_channels))  # 3 stages
```

### 6. DiffusionDiff Family (AI Detection Backbones)

Strictly matching the architecture and checkpoint signature of [`ai-detection-unet-gpu`](https://github.com/khangtruong-ui/ai-detection-unet-gpu):

```python
from jax_model_zoo import (
    DiffusionDiff,
    DiffusionDiffV2,
    DiffusionDiffMinimized,
    DiffusionDiffConfig,
)

# DiffusionDiffMinimized: 2 lines of computation, perpendicular skip fusion, 84-channel Z
cfg = DiffusionDiffConfig(
    image_size=128,
    z_channels=84,
    use_perpendicular_fusion=True,
    num_classes=2,
)
model = DiffusionDiffMinimized(config=cfg)

images = jnp.ones((2, 128, 128, 3))
timesteps = jnp.array([100, 500])
dummy_skips = [
    jnp.ones((2, 64, 64, 32)),
    jnp.ones((2, 32, 32, 32)),
    jnp.ones((2, 16, 16, 32)),
    jnp.ones((2, 8, 8, 64)),
]

params = model.init(jax.random.PRNGKey(0), images, dummy_skips, timesteps)
mask_logits, cls_logits = model.apply(params, images, dummy_skips, timesteps)

print("Segmentation Mask Logits:", mask_logits.shape)  # [2, 128, 128, 1]
print("Auxiliary Classification Logits:", cls_logits.shape)  # [2, 2]
```

---

## Device-Topology-Invariant Checkpointing

Standard Orbax checkpoints can fail when restoring across different device configurations (e.g., training on 8 GPUs and evaluating on 1 GPU or CPU). `jax-model-zoo` solves this by decoupling parameter values from mesh layouts:

```python
from jax_model_zoo import (
    save_checkpoint,
    load_checkpoint,
    save_safetensors,
    load_safetensors,
    DeviceAgnosticCheckpointer,
)

# 1. Safetensors (Universal & Zero-Copy)
save_safetensors(params, "weights.safetensors")
restored = load_safetensors("weights.safetensors")

# 2. Topology-Agnostic Checkpointer (Supports Safetensors, Msgpack, and Orbax)
checkpointer = DeviceAgnosticCheckpointer(format="safetensors")
checkpointer.save(params, "checkpoints/step_1000")
restored_params = checkpointer.restore("checkpoints/step_1000")
```

---

## Bidirectional PyTorch $\leftrightarrow$ JAX Weight Conversion

```python
import torch
from jax_model_zoo import convert_torch_state_dict_to_jax, convert_jax_params_to_torch, ViTConfig

cfg = ViTConfig(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)

# Convert PyTorch state dict to Flax Linen nested dictionary
torch_state_dict = torch.load("torch_vit.pt", map_location="cpu")
jax_params = convert_torch_state_dict_to_jax(torch_state_dict, cfg, model_type="vit")

# Convert back from Flax Linen to PyTorch state dict
torch_state_back = convert_jax_params_to_torch(jax_params, cfg, model_type="vit")
```

---

## Model Surgery

### 1. LoRA Injection and Merging

```python
from jax_model_zoo import inject_lora, merge_lora

# Inject LoRA adapters (rank 8, alpha 16.0) into Attention projections
lora_params = inject_lora(
    params,
    target_patterns=["q_proj", "v_proj"],
    rank=8,
    alpha=16.0,
    rng=jax.random.PRNGKey(42),
)

# Merge LoRA weights back into the base parameters for zero-latency inference
merged_params = merge_lora(lora_params)
```

### 2. Classification Head Replacement

```python
from jax_model_zoo import replace_head

# Replace 1000-class head with a new 10-class head
new_head_params = {
    "kernel": jax.random.normal(jax.random.PRNGKey(1), (768, 10)) * 0.02,
    "bias": jnp.zeros((10,)),
}
custom_params = replace_head(params, "head", new_head_params)
```

### 3. Freezing Parameters for Optax

```python
from jax_model_zoo import get_trainable_param_mask
import optax

# Freeze entire backbone, train only LoRA parameters
mask = get_trainable_param_mask(
    lora_params,
    trainable_patterns=["lora_a", "lora_b"],
    frozen_patterns=["kernel", "bias"],
)

# Use Optax masked optimizer
optimizer = optax.masked(optax.adamw(1e-4), mask)
```

---

## FLOPs & Compute Cost Profiling

Calculate theoretical compute costs directly from traced Jaxprs:

```python
from jax_model_zoo import export_flops, count_parameters

x = jnp.ones((1, 224, 224, 3))
report = export_flops(lambda inp: model.apply(params, inp), x, params=params)

print(f"Total FLOPs: {report.total_flops:,}")
print(f"Total GMACs: {report.gmacs:.2f} GMACs")
print(f"Total Parameters: {report.param_count:,}")
print(report.summary())
```

---

## Multi-Device Scaling (Single-Host GPU / TPU)

Scale forward passes across multiple devices on a single host with `pmap`:

```python
from jax_model_zoo import (
    get_device_count,
    replicate,
    shard_batch,
    unreplicate,
    create_data_parallel_apply,
)

num_devices = get_device_count()
print(f"Found {num_devices} local devices.")

# Replicate parameters across available devices
replicated_params = replicate(params)

# Shard input batches evenly across devices
batched_images = jnp.ones((num_devices * 4, 224, 224, 3))
sharded_x = shard_batch(batched_images)

# Execute data-parallel forward pass
parallel_apply = create_data_parallel_apply(model.apply)
sharded_outputs = parallel_apply(replicated_params, sharded_x)
```

---

## Running Tests

All unit tests enforce strict execution timeouts to prevent runaway processes:

```bash
pytest -v --timeout=120
```

---

## License

Apache License 2.0. See LICENSE for details.
