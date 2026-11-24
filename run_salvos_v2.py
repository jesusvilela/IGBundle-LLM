"""
Neural Glass Salvo Runner v2
=============================
1. Loads Qwen2.5-7B (NF4) + IGBundle geometric adapter (cp-3001)
2. Hyperparameter search on 3 calibration questions with known answers
3. Runs all 15 salvos individually with best params
4. Saves scored results

Uses the same adapter hook integration as app_neural_glass.py.
"""

import os
import sys
import gc
import json
import time
import argparse
import torch
import numpy as np

# Fix Windows charmap crash
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# IGBundle imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from model_state import ModelState

# ── Config ────────────────────────────────────────────────────────────
MODEL_STATE = ModelState()

BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"

# Dynamic checkpoint resolution
_state = MODEL_STATE.load()
ADAPTER_WEIGHTS = _state["checkpoint"]["path"]
SYSTEM_PROMPT = _state["system_prompt"]


# ── Model Loading ─────────────────────────────────────────────────────

def load_model():
    """Load Qwen2.5-7B (NF4) + IGBundle geometric adapter with hook injection."""
    print("=" * 60)
    print(f"Loading Qwen2.5-7B (NF4) + IGBundle adapter")
    print(f"  Checkpoint: {ADAPTER_WEIGHTS}")
    print(f"  {MODEL_STATE.summary()}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    max_memory = {0: "5GiB", "cpu": "30GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config,
        device_map="auto", max_memory=max_memory,
        trust_remote_code=True,
    )

    # Create and load geometric adapter
    config = IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True,
        supported_modalities=["vision", "text"],
        enable_meta_cognition=True,
    )
    adapter = create_geometric_adapter(config).to("cuda")

    if os.path.exists(ADAPTER_WEIGHTS):
        print(f"Loading adapter weights: {ADAPTER_WEIGHTS}", flush=True)
        adapter.load_state_dict(torch.load(ADAPTER_WEIGHTS, map_location="cuda"), strict=False)
        print("Adapter loaded.", flush=True)
    else:
        print(f"WARNING: No adapter weights at {ADAPTER_WEIGHTS}", flush=True)

    # Inject adapter hook (same logic as app_neural_glass.py)
    print("Injecting adapter hook...", flush=True)
    _inject_hook(model, adapter)

    model.eval()
    print("Model ready.", flush=True)
    return model, tokenizer


def _inject_hook(model, adapter):
    """Inject geometric adapter at layer 12 with dtype-safe conversion."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        layers = model.base_model.model.layers
    else:
        print("ERROR: Cannot find transformer layers. Running without adapter hook.")
        return

    target = layers[12]
    original_forward = target.forward

    def hook(hidden_states, *args, **kwargs):
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype

        h_in = h.to("cuda").to(torch.float32)
        with torch.no_grad():
            adapted_out, _ = adapter(h_in, pixel_values=None)

        adapted = adapted_out.to(orig_dtype)
        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted

    target.forward = hook
    print("Adapter hook injected at layer 12.")


# ── Generation ────────────────────────────────────────────────────────

def generate_one(model, tokenizer, prompt: str,
                 temperature=0.7, top_p=0.9, top_k=40,
                 repetition_penalty=1.15, max_new_tokens=2048) -> str:
    """Generate a single response for one prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[-1]

    remaining = 32768 - input_len - 64
    safe_max = max(256, min(max_new_tokens, remaining))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=safe_max,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_tokens = outputs[0][input_len:]
    text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # Cleanup
    del outputs, inputs
    torch.cuda.empty_cache()
    gc.collect()

    return text, len(response_tokens)


# ── Hyperparameter Search ─────────────────────────────────────────────

CALIBRATION = [
    {
        "prompt": "Maria has 3x as many apples as Joao. After Joao buys 12 more and Maria gives away a third of hers, they have the same amount. How many did Maria start with?",
        "answer": "36",
        "check_words": ["36"],
        "anti_words": ["48", "24", "72"],
    },
    {
        "prompt": "If all Bloops are Razzles, and all Razzles are Lazzles, but no Lazzles are Wazzles, what can we conclude about Bloops and Wazzles?",
        "answer": "No Bloops are Wazzles",
        "check_words": ["no", "bloops", "wazzles"],
        "anti_words": [],
    },
    {
        "prompt": "The element with atomic number 79 is gold. It shares a group in the periodic table with the element used in most electrical wiring. What are both elements and what property do they share?",
        "answer": "Gold and Copper, both excellent electrical conductors",
        "check_words": ["gold", "copper"],
        "anti_words": ["lanthanum", "cerium", "silver"],
    },
]

SEARCH_GRID = [
    # (temp, top_p, top_k, rep_penalty) — 12 combinations
    (0.3, 0.85, 30, 1.05),
    (0.3, 0.90, 40, 1.05),
    (0.3, 0.95, 50, 1.0),
    (0.5, 0.85, 30, 1.05),
    (0.5, 0.90, 40, 1.05),
    (0.5, 0.95, 50, 1.0),
    (0.5, 0.90, 40, 1.10),
    (0.7, 0.85, 40, 1.05),
    (0.7, 0.90, 40, 1.05),
    (0.7, 0.95, 50, 1.0),
    (0.85, 0.90, 40, 1.05),
    (0.85, 0.95, 50, 1.10),
]


def score_response(response: str, cal: dict) -> float:
    """Simple scoring: check for expected words, penalize anti-words."""
    lower = response.lower()
    score = 0.0

    # Check for correct answer keywords
    for word in cal["check_words"]:
        if word.lower() in lower:
            score += 1.0 / len(cal["check_words"])

    # Penalize wrong answers
    for word in cal["anti_words"]:
        if word.lower() in lower:
            score -= 0.3

    # Penalize very short or very long responses
    word_count = len(response.split())
    if word_count < 20:
        score -= 0.2
    if word_count > 500:
        score -= 0.1

    # Check for repetition (repeating the same sentence)
    sentences = response.split(".")
    if len(sentences) > 3:
        unique_ratio = len(set(s.strip().lower() for s in sentences if s.strip())) / len(sentences)
        if unique_ratio < 0.5:
            score -= 0.5

    return max(0.0, min(1.0, score))


def hyperparam_search(model, tokenizer):
    """Test generation params on calibration questions."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH")
    print("=" * 60)

    results = []
    for i, (temp, top_p, top_k, rep_pen) in enumerate(SEARCH_GRID):
        config_str = f"T={temp} P={top_p} K={top_k} R={rep_pen}"
        print(f"\n--- Config {i+1}/{len(SEARCH_GRID)}: {config_str} ---", flush=True)

        total_score = 0.0
        for cal in CALIBRATION:
            t0 = time.time()
            response, n_tokens = generate_one(
                model, tokenizer, cal["prompt"],
                temperature=temp, top_p=top_p, top_k=top_k,
                repetition_penalty=rep_pen, max_new_tokens=512,
            )
            elapsed = time.time() - t0
            s = score_response(response, cal)
            total_score += s
            print(f"  Q: {cal['prompt'][:60]}...", flush=True)
            print(f"  A: {response[:150]}...", flush=True)
            print(f"  Score: {s:.2f} | {n_tokens} tok | {elapsed:.1f}s", flush=True)

        avg_score = total_score / len(CALIBRATION)
        results.append((avg_score, temp, top_p, top_k, rep_pen))
        print(f"  >> Average: {avg_score:.3f}")

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 60)
    print("SEARCH RESULTS (sorted)")
    print("=" * 60)
    for rank, (score, t, p, k, r) in enumerate(results):
        marker = " <<<" if rank == 0 else ""
        print(f"  #{rank+1}: score={score:.3f} | T={t} P={p} K={k} R={r}{marker}")

    best = results[0]
    print(f"\nBEST: T={best[1]} P={best[2]} K={best[3]} R={best[4]} (score={best[0]:.3f})")
    return {"temperature": best[1], "top_p": best[2], "top_k": best[3],
            "repetition_penalty": best[4]}


# ── Salvos ────────────────────────────────────────────────────────────

SALVOS = {
    "Salvo 1 - Abstract Reasoning": [
        "A sequence follows this rule: each term is the sum of the digits of the previous term squared. Starting from 39, what are the next 5 terms?",
        "If all Bloops are Razzles, and all Razzles are Lazzles, but no Lazzles are Wazzles, what can we conclude about Bloops and Wazzles?",
        "A 4x4 grid has numbers 1-16 placed so each row, column, and 2x2 quadrant sums to the same value. What is that value, and what constraint does it place on the center 2x2?",
    ],
    "Salvo 2 - Multi-step Math": [
        "Maria has 3x as many apples as Joao. After Joao buys 12 more and Maria gives away a third of hers, they have the same amount. How many did Maria start with?",
        "A train travels from A to B at 80 km/h and returns at 120 km/h. What is the average speed for the round trip, and why is it not 100 km/h?",
        "If you invest 1,000 euros at 5% annual compound interest, how many years until it doubles? Give the exact formula and an approximation.",
    ],
    "Salvo 3 - MMLU Knowledge": [
        "What is the difference between a tensor, a vector, and a scalar in the context of differential geometry? Give a concrete example of each.",
        "Explain why the Fisher information matrix is always positive semi-definite. What does it mean when it's singular?",
        "What is the second law of thermodynamics, and how does it relate to the arrow of time? Name one system that appears to violate it and explain why it doesn't.",
    ],
    "Salvo 4 - Multi-hop Reasoning": [
        "The philosopher who wrote the Critique of Pure Reason was born in the same city where he died. That city is now in which country, and what language is spoken there today?",
        "The element with atomic number 79 shares a group in the periodic table with the element used in most electrical wiring. What are both elements and what property do they share?",
        "The author of Brave New World was the grandson of a scientist famous for defending a controversial 19th century theory. What was that theory, and who originally proposed it?",
    ],
    "Salvo 5 - Stress Tests": [
        "Describe consciousness from the perspective of information theory, thermodynamics, and differential geometry simultaneously. Where do the three frameworks agree?",
        "Write a step-by-step proof that there are infinitely many primes, then explain the proof as if speaking to a 10-year-old, then to a mathematics PhD student.",
        "What would a valid counterexample to the Riemann Hypothesis look like? Why would finding one be catastrophic for number theory?",
    ],
}


def run_salvos(model, tokenizer, params: dict, max_tokens: int = 4096):
    """Run all 15 salvos individually with best params."""
    print("\n" + "=" * 60)
    print(f"RUNNING SALVOS (params: {params})")
    print("=" * 60)

    os.makedirs("salvos_results", exist_ok=True)
    step = MODEL_STATE.state["checkpoint"].get("step", "unknown")
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"salvos_results/probe_results_v2_cp{step}_{ts}.md"

    prompt_count = 0
    total = sum(len(v) for v in SALVOS.values())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Neural Glass Salvo Probe v2 — cp-3001\n\n")
        f.write(f"**Params:** T={params['temperature']} P={params['top_p']} "
                f"K={params['top_k']} R={params['repetition_penalty']}\n\n")
        f.write(f"**Model:** Qwen2.5-7B (NF4) + IGBundle Geometric Adapter (cp-3001)\n\n")
        f.write("---\n\n")

        for salvo_name, prompts in SALVOS.items():
            print(f"\n{'─'*40}")
            print(f"  {salvo_name}")
            print(f"{'─'*40}")
            f.write(f"## {salvo_name}\n\n")

            for prompt in prompts:
                prompt_count += 1
                print(f"\n[{prompt_count}/{total}] {prompt[:80]}...", flush=True)
                f.write(f"**Prompt:** {prompt}\n\n")

                t0 = time.time()
                try:
                    response, n_tokens = generate_one(
                        model, tokenizer, prompt,
                        max_new_tokens=max_tokens,
                        **params,
                    )
                    elapsed = time.time() - t0

                    # Get VRAM usage
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated() / (1024**3)
                        vram = f"VRAM: {alloc:.2f}GB"
                    else:
                        vram = "N/A"

                    print(f"  >> {n_tokens} tokens | {elapsed:.1f}s | {vram}", flush=True)
                    print(f"  >> {response[:200]}...", flush=True)

                    f.write(f"**Response ({n_tokens} tokens, {elapsed:.1f}s):**\n\n")
                    f.write(f"{response}\n\n---\n\n")

                except torch.cuda.OutOfMemoryError:
                    print(f"  CUDA OOM! Clearing and continuing...")
                    f.write("**Response:** [CUDA OOM — skipped]\n\n---\n\n")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"  Error: {e}")
                    f.write(f"**Response:** [Error: {e}]\n\n---\n\n")
                    torch.cuda.empty_cache()
                    gc.collect()

    print(f"\nResults saved to {output_file}")
    return output_file


# ── Main ──────────────────────────────────────────────────────────────

def score_salvo_results(output_file: str) -> float:
    """Parse salvo results file and compute mean score using calibration-style checks."""
    # Simple heuristic: count how many salvos got reasonable-length responses
    # without repetition or errors. Full scoring is done by human review.
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Count successful responses (not OOM, not Error, not empty)
        total = content.count("**Prompt:**")
        errors = content.count("[CUDA OOM") + content.count("[Error:")
        completed = total - errors
        return round(completed / max(total, 1), 4)
    except Exception:
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Glass Salvo Runner v2")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip hyperparam search, use defaults")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per salvo response")
    parser.add_argument("--temp", type=float, default=None,
                        help="Override temperature")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override top_p")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top_k")
    parser.add_argument("--rep-penalty", type=float, default=None,
                        help="Override repetition_penalty")
    args = parser.parse_args()

    # Pre-run: load best params from ModelState if score > 0.7
    state = MODEL_STATE.load()
    evo = state["evolution"]

    model, tokenizer = load_model()

    if args.skip_search or (evo["best_score"] > 0.7 and evo["best_params"]):
        if evo["best_score"] > 0.7 and evo["best_params"]:
            best_params = dict(evo["best_params"])
            print(f"[ModelState] Using evolved params (score={evo['best_score']:.3f}): {best_params}")
        else:
            best_params = {
                "temperature": args.temp or 0.5,
                "top_p": args.top_p or 0.90,
                "top_k": args.top_k or 40,
                "repetition_penalty": args.rep_penalty or 1.05,
            }
            print(f"Using default params: {best_params}")
    else:
        best_params = hyperparam_search(model, tokenizer)

    # Allow CLI overrides after search
    if args.temp is not None:
        best_params["temperature"] = args.temp
    if args.top_p is not None:
        best_params["top_p"] = args.top_p
    if args.top_k is not None:
        best_params["top_k"] = args.top_k
    if args.rep_penalty is not None:
        best_params["repetition_penalty"] = args.rep_penalty

    output = run_salvos(model, tokenizer, best_params, max_tokens=args.max_tokens)
    print(f"\nDone! Results: {output}")
    print(f"Final VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f}GB allocated")

    # Post-run: evolve ModelState with results
    mean_score = score_salvo_results(output)
    print(f"\n[ModelState] Salvo completion score: {mean_score:.4f}")
    best = MODEL_STATE.evolve(
        mean_score=mean_score,
        params=best_params,
        run_detail={
            "checkpoint": state["checkpoint"]["path"],
            "checkpoint_step": state["checkpoint"]["step"],
            "output_file": output,
        }
    )
    print(f"[ModelState] Best score: {best:.4f}")
    print(f"[ModelState] {MODEL_STATE.summary()}")
