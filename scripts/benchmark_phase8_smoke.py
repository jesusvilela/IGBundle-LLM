"""Paired base-vs-Phase-8 geometric-adapter smoke benchmark.

This is a loader and generation-integrity gate, not a full benchmark.  It
preserves raw outputs and provenance for identical greedy-decoding prompts.
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SOURCE_ROOT = r"H:\LLM-MANIFOLD\igbundle-llm"
BASE_MODEL = r"H:\LLM-MANIFOLD\igbundle_qwen7b_cp600"
CHECKPOINT = r"H:\LLM-MANIFOLD\igbundle_phase8_training\checkpoint-3000\adapter_weights.pt"
OUTPUT_DIR = r"H:\LLM-MANIFOLD\benchmark-runs\phase8-smoke-2026-09-14"
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.path.insert(0, os.path.join(SOURCE_ROOT, "src"))
    from igbundle.core.config import IGBundleConfig
    from igbundle.modules.geometric_adapter import create_geometric_adapter

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map="auto",
        max_memory={0: "6GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    base_outputs = [generate(model, tokenizer, prompt) for prompt in PROMPTS]

    config = IGBundleConfig(hidden_size=3584, num_components=8, latent_dim=64,
        num_categories=16, use_dynamics=True, use_geodesic_attn=True,
        vision_dim=1152, supported_modalities=["vision", "text"])
    adapter = create_geometric_adapter(config).to("cuda").eval()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    incompatible = adapter.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint incompatibility: {incompatible}")

    layer = model.model.layers[12]
    def hook(_, __, output):
        hidden = output[0] if isinstance(output, tuple) else output
        adapted, _ = adapter(hidden.to(torch.float32))
        adapted = adapted.to(hidden.dtype)
        return (adapted,) + output[1:] if isinstance(output, tuple) else adapted
    handle = layer.register_forward_hook(hook)
    adapter_outputs = [generate(model, tokenizer, prompt) for prompt in PROMPTS]
    handle.remove()

    record = {
        "kind": "paired generation smoke test; not a task-quality benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "adapter_config": config.to_dict(),
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "samples": [{"prompt": p, "base": b, "phase8_geometric": a} for p, b, a in zip(PROMPTS, base_outputs, adapter_outputs)],
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
