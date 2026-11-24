"""GPU smoke test for Gemma4-width IGBundle geometry-core runtime.

This validates the CUDA adapter/n-manifold/EPIC telemetry path without loading
the full Gemma4 language model. It is intended for Windows systems where the
full safetensor mmap may be blocked by pagefile/commit limits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.telemetry import (
    format_epic_panel,
    format_stack_panel,
    summarize_epic_stack,
    summarize_manifold_stack,
)
import app_neural_glass as app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA IGBundle EPIC telemetry smoke test.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "igbundle_phase9_odyssey" / "checkpoint-3000" / "adapter_weights.pt"),
        help="Geometric adapter checkpoint path.",
    )
    parser.add_argument("--hidden-size", type=int, default=2560, help="Live base-model residual width.")
    parser.add_argument("--tokens", type=int, default=2, help="Synthetic sequence length for the adapter forward pass.")
    parser.add_argument("--batch", type=int, default=1, help="Synthetic batch size for the adapter forward pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return 2

    checkpoint = os.path.abspath(args.checkpoint)
    geometry = app._infer_checkpoint_geometry(checkpoint)
    config = IGBundleConfig(
        hidden_size=args.hidden_size,
        latent_dim=geometry["latent_dim"],
        num_components=geometry["num_components"],
        num_categories=geometry["num_categories"],
        bottleneck_dim=args.hidden_size // 4,
        manifold_type="poincare",
        use_dynamics=True,
        use_geodesic_attn=True,
        supported_modalities=["vision", "text", "audio"],
        enable_meta_cognition=True,
        use_delta_fiber=True,
        delta_mem_dim=64,
        delta_num_heads=4,
    )

    print("gpu_before", torch.cuda.mem_get_info())
    adapter = create_geometric_adapter(config).to("cuda")
    loaded = app._load_geometric_checkpoint_safe(adapter, checkpoint)
    if not loaded:
        print("checkpoint_load_failed")
        return 3

    hidden = torch.randn(args.batch, args.tokens, args.hidden_size, device="cuda", dtype=torch.float32) * 0.02
    with torch.no_grad():
        adapted, geo_state = adapter(hidden)

    telemetry = {
        "constraint_score": 0.82,
        "curvature_status": "gpu-smoke",
        "adapter_provenance": {"mode": "gemma4-teacher-geometry-core", "checkpoint_path": checkpoint},
    }
    n_stack = summarize_manifold_stack(
        adapter,
        geo_state,
        hidden_state=hidden,
        curvature=1.0,
        constraint_score=telemetry["constraint_score"],
    )
    telemetry["n_stack"] = n_stack
    epic_stack = summarize_epic_stack(adapter, geo_state, telemetry, memory=object())

    print("geometry", geometry)
    print("adapted_shape", tuple(adapted.shape), "device", adapted.device.type)
    print("n_stack_panel")
    print(format_stack_panel(n_stack))
    print("epic_panel")
    print(format_epic_panel(epic_stack))
    print("gpu_after", torch.cuda.mem_get_info())

    assert adapted.is_cuda
    assert n_stack["dims"]["latent_dim"] == geometry["latent_dim"]
    assert n_stack["dims"]["components"] == geometry["num_components"]
    assert n_stack["dims"]["categories"] == geometry["num_categories"]
    assert "self_reflection" in n_stack["quality_vector"]
    assert epic_stack["present_count"] >= 14
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
