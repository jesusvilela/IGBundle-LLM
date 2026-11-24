#!/usr/bin/env python
"""
GSP Smoke Test — Does the Geometric Steering Probe actually steer?

Runs 5 benchmark prompts with and without GSP attached, comparing:
  - Entropy S distribution
  - Bundle activation distribution (which fibers fire)
  - Generation quality (degeneration, coherence)
  - New JSD-from-uniform metric

Null hypothesis: Without GSP, S ≈ 0.85 and Bundle-6 locked.
Expected with GSP: S moves toward 1.20, bundle distribution widens.

Usage:
    python run_gsp_smoke_test.py [--checkpoint PATH] [--max-tokens 128]
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Benchmark prompts — diverse domains to test S response
PROMPTS = [
    {
        "prompt": "Explain the difference between correlation and causation with a concrete example.",
        "domain": "reasoning",
    },
    {
        "prompt": "A farmer has 17 sheep. All but 9 die. How many are left?",
        "domain": "math_trick",
    },
    {
        "prompt": "Write a short poem about the color of silence.",
        "domain": "creative",
    },
    {
        "prompt": "What are the main arguments for and against nuclear energy?",
        "domain": "analytical",
    },
    {
        "prompt": "Consider a town where the barber shaves everyone who does not shave themselves. Does the barber shave himself?",
        "domain": "paradox",
    },
]


def load_model_and_adapter(model_id: str, checkpoint_path: str):
    """Load Qwen 7B NF4 + IGBundle adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from igbundle.core.config import IGBundleConfig
    from igbundle.modules.geometric_adapter import create_geometric_adapter

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen 7B NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "30GiB"},
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    print("Creating geometric adapter...")
    config = IGBundleConfig(
        hidden_size=3584,
        num_components=8,
        latent_dim=64,
        num_categories=16,
        use_dynamics=True,
        use_geodesic_attn=True,
        supported_modalities=["vision", "text"],
    )
    adapter = create_geometric_adapter(config).to(DEVICE)

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=DEVICE)
    adapter.load_state_dict(state, strict=False)
    adapter.eval()

    return model, tokenizer, adapter, config


def inject_adapter_hook(model, adapter):
    """Inject adapter at layer 12, return geo_state collector and cleanup handle."""
    geo_states = []

    layers = model.model.layers
    target_layer = layers[12]
    original_forward = target_layer.forward

    def adapter_hook(hidden_states, *args, **kwargs):
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype
        h_in = h.to(DEVICE).to(torch.float32)

        with torch.no_grad():
            adapted_out, geo_state = adapter(h_in)

        geo_states.append(geo_state)

        # Perturbation clamping (15% max ratio)
        delta = adapted_out - h_in
        base_norm = h_in.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_norm = delta.norm(dim=-1, keepdim=True)
        max_delta = 0.15 * base_norm
        scale = torch.where(
            delta_norm > max_delta,
            max_delta / delta_norm,
            torch.ones_like(delta_norm),
        )
        adapted = (h_in + delta * scale).to(orig_dtype)

        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted

    target_layer.forward = adapter_hook
    return geo_states, original_forward


def extract_telemetry(geo_states):
    """Extract S, K, bundle activations, JSD from collected geo states."""
    if not geo_states:
        return {"S": [], "K": [], "bundles": {}, "fiber_jsd": []}

    s_vals = []
    k_vals = []
    bundle_counts = Counter()
    jsd_vals = []

    for state in geo_states:
        # Entropy from fiber sections
        if state.fiber_sections is not None:
            p = state.fiber_sections.clamp(min=1e-8)
            H = -(p * p.log()).sum(dim=-1).mean()
            s_vals.append(H.item())

            # Bundle activation: which component has max total section weight
            # sections shape: (B, T, P, K) — sum over K, mean over B,T to get per-component
            comp_weight = p.sum(dim=-1).mean(dim=(0, 1))  # (P,)
            dominant = comp_weight.argmax().item()
            bundle_counts[dominant] += 1

            # JSD from uniform
            K = p.shape[-1]
            uniform = torch.ones_like(p) / K
            m = 0.5 * (p + uniform)
            kl_p = (p * (p.log() - m.clamp(min=1e-8).log())).sum(dim=-1)
            kl_u = (uniform * (uniform.log() - m.clamp(min=1e-8).log())).sum(dim=-1)
            jsd = 0.5 * (kl_p + kl_u)
            jsd_vals.append(jsd.mean().item())

        # Curvature — not directly in GeometricState; compute from metric if available
        if state.metric is not None and hasattr(state.metric, 'curvature'):
            val = state.metric.curvature
            k_vals.append(val.item() if hasattr(val, "item") else float(val))

    return {
        "S": s_vals,
        "K": k_vals,
        "bundles": dict(bundle_counts),
        "fiber_jsd": jsd_vals,
    }


def run_prompts(model, tokenizer, max_tokens=128):
    """Generate responses for all benchmark prompts."""
    results = []
    for p in PROMPTS:
        messages = [
            {"role": "system", "content": "Answer directly with step-by-step reasoning. Be concise."},
            {"role": "user", "content": p["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = out[0, enc.input_ids.shape[-1] :]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(
            {
                "domain": p["domain"],
                "prompt": p["prompt"],
                "response": response[:500],
                "tokens": len(gen_ids),
            }
        )
        torch.cuda.empty_cache()

    return results


def mean_or(lst, default=0.0):
    return sum(lst) / len(lst) if lst else default


def main():
    parser = argparse.ArgumentParser(description="GSP Smoke Test")
    parser.add_argument(
        "--model-id", default="h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
    )
    parser.add_argument(
        "--checkpoint",
        default="igbundle_phase9_odyssey/checkpoint-3000/adapter_weights.pt",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer, adapter, config = load_model_and_adapter(
        args.model_id, args.checkpoint
    )

    # ── Phase 1: WITHOUT GSP ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: WITHOUT GSP (baseline)")
    print("=" * 60)

    geo_states_no_gsp, original_fwd = inject_adapter_hook(model, adapter)
    results_no_gsp = run_prompts(model, tokenizer, args.max_tokens)
    telemetry_no_gsp = extract_telemetry(geo_states_no_gsp)

    for r in results_no_gsp:
        print(f"\n[{r['domain']}] ({r['tokens']} tok)")
        print(f"  {r['response'][:200]}")

    s_no = telemetry_no_gsp["S"]
    jsd_no = telemetry_no_gsp["fiber_jsd"]
    print(f"\n--- Baseline Telemetry ---")
    print(f"  S:       mean={mean_or(s_no):.4f}  (n={len(s_no)})")
    print(f"  JSD:     mean={mean_or(jsd_no):.4f}")
    print(f"  Bundles: {telemetry_no_gsp['bundles']}")
    if telemetry_no_gsp["K"]:
        print(f"  K (first 5): {telemetry_no_gsp['K'][:5]}")

    # Restore original forward for clean Phase 2
    model.model.layers[12].forward = original_fwd
    geo_states_no_gsp.clear()
    torch.cuda.empty_cache()

    # ── Phase 2: WITH GSP ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: WITH GSP (S_target=1.20)")
    print("=" * 60)

    geo_states_gsp, original_fwd_2 = inject_adapter_hook(model, adapter)

    # Create and attach GSP with initial telemetry reflecting baseline
    from igbundle.steering.gsp import create_gsp_for_qwen7b

    telemetry_live = {
        "curvature": mean_or(telemetry_no_gsp["K"], -5.0),
        "entropy": mean_or(s_no, 0.85),
        "active_fiber": "Bundle-6",
        "constraint_score": 1.0,
    }
    gsp, bridge = create_gsp_for_qwen7b(telemetry_dict=telemetry_live)
    gsp.attach(model, range(8, 21))
    bridge.poll_and_update()

    results_gsp = run_prompts(model, tokenizer, args.max_tokens)
    telemetry_gsp = extract_telemetry(geo_states_gsp)

    for r in results_gsp:
        print(f"\n[{r['domain']}] ({r['tokens']} tok)")
        print(f"  {r['response'][:200]}")

    s_gsp = telemetry_gsp["S"]
    jsd_gsp = telemetry_gsp["fiber_jsd"]
    print(f"\n--- GSP Telemetry ---")
    print(f"  S:       mean={mean_or(s_gsp):.4f}  (n={len(s_gsp)})")
    print(f"  JSD:     mean={mean_or(jsd_gsp):.4f}")
    print(f"  Bundles: {telemetry_gsp['bundles']}")
    if telemetry_gsp["K"]:
        print(f"  K (first 5): {telemetry_gsp['K'][:5]}")

    # Cleanup
    gsp.detach()
    model.model.layers[12].forward = original_fwd_2

    # ── Comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: NO GSP vs GSP")
    print("=" * 60)

    s_no_m = mean_or(s_no)
    s_gsp_m = mean_or(s_gsp)
    jsd_no_m = mean_or(jsd_no)
    jsd_gsp_m = mean_or(jsd_gsp)

    print(f"  S mean:    {s_no_m:.4f} -> {s_gsp_m:.4f}   D={s_gsp_m - s_no_m:+.4f}")
    print(f"  JSD mean:  {jsd_no_m:.4f} -> {jsd_gsp_m:.4f}   D={jsd_gsp_m - jsd_no_m:+.4f}")
    print(f"  Bundles:   {telemetry_no_gsp['bundles']} -> {telemetry_gsp['bundles']}")

    # Verdict
    s_moved = abs(s_gsp_m - s_no_m) > 0.05
    bundle_changed = telemetry_no_gsp["bundles"] != telemetry_gsp["bundles"]
    print(f"\n  S moved?       {'YES' if s_moved else 'NO'} (threshold: D>0.05)")
    print(f"  Bundle shift?  {'YES' if bundle_changed else 'NO'}")
    if s_moved:
        print("  -> GSP IS STEERING entropy. Hooks are firing, PID is active.")
    else:
        print("  -> GSP did NOT steer S. Investigate: hooks not firing or PID gains insufficient.")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint,
        "no_gsp": {
            "responses": results_no_gsp,
            "S_mean": s_no_m,
            "S_values": s_no,
            "bundles": telemetry_no_gsp["bundles"],
            "jsd_mean": jsd_no_m,
            "K_values": telemetry_no_gsp["K"][:10],
        },
        "with_gsp": {
            "responses": results_gsp,
            "S_mean": s_gsp_m,
            "S_values": s_gsp,
            "bundles": telemetry_gsp["bundles"],
            "jsd_mean": jsd_gsp_m,
            "K_values": telemetry_gsp["K"][:10],
        },
        "verdict": {
            "S_moved": s_moved,
            "S_delta": s_gsp_m - s_no_m,
            "bundle_changed": bundle_changed,
        },
    }
    outfile = f"gsp_smoke_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
