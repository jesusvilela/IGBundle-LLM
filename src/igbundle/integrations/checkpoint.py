"""Fail-closed restoration of the default adapter, not checkpoint provenance."""

from collections.abc import Mapping
from pathlib import Path

import torch

from ..modules.adapter import IGBundleAdapter


def require_checkpoint_file(path):
    """Check availability before callers allocate a base model."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required adapter checkpoint is missing: {path}")
    return path


def restore_adapter_state(model, state, *, full_state=False):
    """Validate all adapter tensors before copying any; never rewrite key prefixes.

    Adapter-only mappings must contain exactly the registered default adapters'
    keys. Full-state mappings may additionally contain known model keys, which
    are ignored, not restored. This does not restore LoRA or certify base identity.
    """
    adapters = {
        name: module for name, module in model.named_modules()
        if isinstance(module, IGBundleAdapter)
    }
    if not adapters:
        raise ValueError("No default IGBundle adapters are registered")
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise ValueError("Checkpoint must be a string-keyed tensor state dictionary")

    expected = {}
    for name, module in adapters.items():
        prefix = f"{name}." if name else ""
        for key, target in module.state_dict().items():
            expected[prefix + key] = target

    missing = expected.keys() - state.keys()
    allowed = model.state_dict().keys() if full_state else expected.keys()
    unexpected = state.keys() - allowed
    if missing or unexpected:
        raise ValueError(
            f"Adapter coverage mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    prepared = {}
    for key, target in expected.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.is_meta
                or value.layout != torch.strided or value.is_quantized):
            raise ValueError(f"{key}: expected a materialized dense tensor")
        if value.shape != target.shape:
            raise ValueError(f"{key}: shape {tuple(value.shape)} != {tuple(target.shape)}")
        if target.is_meta:
            raise ValueError(f"{key}: destination adapter is still on the meta device")
        if value.dtype != target.dtype and not (
            value.is_floating_point() and target.is_floating_point()
        ):
            raise ValueError(f"{key}: incompatible dtype {value.dtype} -> {target.dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key}: nonfinite checkpoint values")
        converted = value.detach().to(device="cpu", dtype=target.dtype)
        if not torch.isfinite(converted).all():
            raise ValueError(f"{key}: conversion to {target.dtype} overflows")
        prepared[key] = converted

    for name, module in adapters.items():
        prefix = f"{name}." if name else ""
        module.load_state_dict(
            {key: prepared[prefix + key] for key in module.state_dict()},
            strict=True,
        )
    return len(prepared)


def load_weights(model, path, *, full_state=False):
    """Read tensor-only state on CPU; do not retry unsafe pickle loading."""
    path = require_checkpoint_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    return restore_adapter_state(model, state, full_state=full_state)
