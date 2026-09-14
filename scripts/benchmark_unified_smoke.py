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
import ctypes
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = r"H:\LLM-MANIFOLD\igbundle-llm"
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "igbundle_unified_deployment", "src")
BASE_MODEL = r"H:\LLM-MANIFOLD\igbundle_qwen7b_cp600"
CHECKPOINT = os.path.join(PROJECT_ROOT, "igbundle_unified_training", "final", "adapter_weights.pt")
OUTPUT_DIR = r"H:\LLM-MANIFOLD\benchmark-runs\unified-smoke-2026-09-14"
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "offload_unified_smoke")
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


def windows_available_commit_bytes():
    """Return available Windows commit/pagefile bytes when available."""
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPageFile)


def model_safetensor_bytes(model_path):
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(model_path)
        for name in files if name.endswith(".safetensors")
    )


def load_qwen_neural_glass_style():
    """Load Qwen with Neural Glass's explicit GPU/CPU split and disk offload."""
    available_commit = windows_available_commit_bytes()
    shard_bytes = model_safetensor_bytes(BASE_MODEL)
    required_commit = shard_bytes + 2 * 1024 ** 3
    if available_commit is not None and available_commit < required_commit:
        raise RuntimeError(
            "Insufficient Windows commit headroom for safe 7B load: "
            f"available={available_commit / 1024 ** 3:.2f} GiB, "
            f"required~={required_commit / 1024 ** 3:.2f} GiB "
            f"({shard_bytes / 1024 ** 3:.2f} GiB shards + 2 GiB headroom). "
            "Increase the paging file or close memory-heavy applications, then rerun."
        )
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep the adapted layer and language-model boundary on GPU; offload the
    # upper half. This mirrors Neural Glass's explicit placement discipline.
    device_map = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0}
    for index in range(28):
        device_map[f"model.layers.{index}"] = 0 if index <= 13 else "cpu"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quantization, device_map=device_map,
        max_memory={0: "5GiB", "cpu": "24GiB"}, offload_folder=OFFLOAD_DIR,
        offload_state_dict=True, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    return model.eval(), tokenizer


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

    model, tokenizer = load_qwen_neural_glass_style()

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
