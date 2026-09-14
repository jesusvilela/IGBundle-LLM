"""Reproducible paired smoke benchmark for the exact unified checkpoint.

This is deliberately a small generation-integrity gate, not a quality claim.
It compares identical greedy-decoding samples with and without the adapter,
records all raw text and provenance, and rejects incompatible or inert weights.
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = r"H:\LLM-MANIFOLD\igbundle-llm"
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "igbundle_unified_deployment", "src")
BASE_MODEL = r"H:\LLM-MANIFOLD\igbundle_qwen7b_cp600"
CHECKPOINT = os.path.join(PROJECT_ROOT, "igbundle_unified_training", "final", "adapter_weights.pt")
OUTPUT_DIR = r"H:\LLM-MANIFOLD\benchmark-runs\unified-smoke-2026-09-14"
PROMPTS = [
    "What is the next number in the sequence 2, 6, 12, 20, 30? Answer briefly.",
    "If every A is B and no B is C, can any A be C? Answer briefly.",
    "A rectangle is 3 by 4. What is its diagonal length? Answer briefly.",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def validate_adapter(adapter):
    """Reject a checkpoint that is non-finite, inert, or implausibly disruptive."""
    torch.manual_seed(7)
    hidden = torch.randn(1, 4, 3584, device="cuda")
    with torch.enable_grad():
        adapted, _ = adapter(hidden, return_aux=True)
    relative_delta = float((adapted - hidden).norm() / hidden.norm())
    if not bool(torch.isfinite(adapted).all()):
        raise RuntimeError("Adapter probe produced non-finite values")
    if relative_delta == 0.0:
        raise RuntimeError("Adapter probe is inert (zero residual)")
    if relative_delta > 0.25:
        raise RuntimeError(f"Adapter probe is too disruptive: relative_delta={relative_delta:.6f}")
    return relative_delta


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.path.insert(0, SOURCE_ROOT)
    from igbundle.core.config import IGBundleConfig
    from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=8192, dtype=None,
        load_in_4bit=True, trust_remote_code=True, device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True, vision_dim=1152,
        supported_modalities=["vision", "text"],
    )
    adapter = GeometricIGBundleAdapter(config).to("cuda").eval()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)
    relative_delta = validate_adapter(adapter)

    base_outputs = [generate(model, tokenizer, prompt) for prompt in PROMPTS]
    layer = model.model.layers[12]

    def hook(_, __, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.enable_grad():
            adapted, _ = adapter(hidden.to(torch.float32), return_aux=True)
        adapted = adapted.to(hidden.dtype)
        return (adapted,) + output[1:] if isinstance(output, tuple) else adapted

    handle = layer.register_forward_hook(hook)
    try:
        adapter_outputs = [generate(model, tokenizer, prompt) for prompt in PROMPTS]
    finally:
        handle.remove()

    record = {
        "kind": "paired generation smoke test; not a task-quality benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "adapter_source": SOURCE_ROOT,
        "adapter_config": config.to_dict(),
        "adapter_probe_relative_delta": relative_delta,
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "samples": [{"prompt": p, "base": b, "unified_geometric": a} for p, b, a in zip(PROMPTS, base_outputs, adapter_outputs)],
    }
    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
