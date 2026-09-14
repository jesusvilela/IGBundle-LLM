"""Numerical integrity benchmark for a geometric adapter checkpoint.

This does not measure language-task quality. It verifies that an exact source /
checkpoint pair has a stable, non-zero residual across deterministic hidden-state
shapes, and stores the raw measurements for later comparison.
"""
import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, help="Source snapshot containing igbundle/.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this adapter-integrity run")
    sys.path.insert(0, os.path.abspath(args.source_root))
    config_module = importlib.import_module("igbundle.core.config")
    adapter_module = importlib.import_module("igbundle.modules.geometric_adapter")
    config = config_module.IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True, vision_dim=1152,
        supported_modalities=["vision", "text"],
    )
    adapter = adapter_module.GeometricIGBundleAdapter(config).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)

    cases = [(1, 1), (1, 4), (2, 8), (2, 32)]
    measurements = []
    for seed, (batch, tokens) in enumerate(cases, start=7):
        torch.manual_seed(seed)
        hidden = torch.randn(batch, tokens, 3584, device="cuda")
        # The checkpoint's Hamiltonian step computes an internal derivative.
        # Do not wrap this call in inference_mode(), which disables that path.
        with torch.enable_grad():
            adapted, geometric_state = adapter(hidden, return_aux=True)
        measurements.append({
            "seed": seed,
            "input_shape": list(hidden.shape),
            "finite": bool(torch.isfinite(adapted).all()),
            "relative_residual_norm": float((adapted - hidden).norm() / hidden.norm()),
            "base_coordinates_shape": list(geometric_state.base_coordinates.shape),
            "fiber_sections_shape": list(geometric_state.fiber_sections.shape),
        })

    if not all(row["finite"] for row in measurements):
        raise RuntimeError("Non-finite output found")
    if any(row["relative_residual_norm"] == 0.0 for row in measurements):
        raise RuntimeError("Inert residual found")
    record = {
        "kind": "adapter numerical-integrity benchmark; not a language-task benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": os.path.abspath(args.source_root),
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        },
        "measurements": measurements,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
