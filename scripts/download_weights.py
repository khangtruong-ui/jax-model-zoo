#!/usr/bin/env python3
"""
Download online pretrained weights from reputable sources (HuggingFace Hub, timm, official repos)
and convert them to topology-agnostic JAX/Flax Linen parameter checkpoints.

Usage examples:
    python scripts/download_weights.py --model vit-b
    python scripts/download_weights.py --model all --output-dir ./checkpoints
    python scripts/download_weights.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_model_zoo.core.registry import (
    PRETRAINED_CATALOG,
    canonicalize_model_name,
    download_and_convert,
    get_model_info,
    list_models,
)
from jax_model_zoo.core.flops import count_parameters


def print_catalog():
    print("================================================================================")
    print("                   JAX MODEL ZOO - PRETRAINED ONLINE CATALOG                    ")
    print("================================================================================")
    for name, info in PRETRAINED_CATALOG.items():
        aliases = ", ".join(info.get("aliases", []))
        print(f"Model ID:      {name}")
        print(f"Source URL:    {info['online_source']} ({info['source_type']})")
        print(f"Description:   {info['description']}")
        print(f"Input Shape:   {info.get('input_shape')}")
        print(f"Aliases:       {aliases}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download online pretrained checkpoints and convert to JAX/Flax Linen format."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit-b",
        help="Model ID or comma-separated list of IDs to download (or 'all').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the converted JAX checkpoints.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "msgpack", "orbax"],
        help="Checkpoint format for device-topology-agnostic storage.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification forward pass.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in catalog and exit.",
    )

    args = parser.parse_args()

    if args.list:
        print_catalog()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model.lower() == "all":
        # Exclude models without online weights (like diffusion-diff)
        target_models = [m for m, info in PRETRAINED_CATALOG.items() if info.get("has_pretrained_online", True)]
    else:
        target_models = [canonicalize_model_name(m.strip()) for m in args.model.split(",")]

    print(f"Downloading and converting {len(target_models)} model(s) to {output_dir.resolve()}...")
    success_count = 0

    for model_name in target_models:
        try:
            info = get_model_info(model_name)
            if not info.get("has_pretrained_online", True):
                print(f"[SKIP] Model '{model_name}' does not have public online pretrained weights.")
                continue

            print(f"\n>>> Processing [{model_name}] from {info['online_source']}...")
            start_t = time.time()
            model, params, dest = download_and_convert(
                model_name,
                output_dir=output_dir,
                format=args.format,
                verify=not args.no_verify,
            )
            elapsed = time.time() - start_t
            num_params = count_parameters(params)
            file_size_mb = dest.stat().st_size / (1024 * 1024) if dest.is_file() else 0.0

            print(f"    [SUCCESS] Saved to: {dest}")
            print(f"    Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
            print(f"    File Size:  {file_size_mb:.2f} MB")
            print(f"    Time Taken: {elapsed:.2f}s")
            success_count += 1
        except Exception as e:
            print(f"    [ERROR] Failed to process {model_name}: {e}")

    print(f"\nDone! Successfully processed {success_count}/{len(target_models)} models.")


if __name__ == "__main__":
    main()
