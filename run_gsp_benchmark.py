"""
GSP CMA-ES Benchmark Harness
═════════════════════════════
Black-box optimization of the Geometric Steering Probe genome (~20 scalars)
using evaluation fitness over a curated prompt battery.

Fitness = weighted combination of:
  1. EOS rate:         fraction of generations emitting <|im_end|> naturally
  2. Quality score:    keyword/constraint matching on calibration prompts
  3. Entropy health:   S in target band [0.8, 1.4] (not frozen, not collapsed)
  4. Degeneration:     penalize repetition loops, prompt leakage
  5. Curvature drift:  K shouldn't change drastically from baseline

Operates on a pre-loaded model — one load, many genome evaluations.
Each genome eval runs a subset of prompts (configurable, default 8).

Usage:
    python run_gsp_benchmark.py                     # default: 30 generations, popsize 8
    python run_gsp_benchmark.py --generations 50    # longer run
    python run_gsp_benchmark.py --prompts 15        # full salvo battery per eval
"""

import os
import sys
import gc
import json
import time
import zlib
import argparse
import torch
import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.steering.gsp import (
    create_gsp_for_qwen7b, GSPGenome, GSPEvolver, GeometricSteeringProbe,
)
from model_state import ModelState


# ═════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════

BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512          # short budget — we care about EOS and quality, not length
GEN_TEMPERATURE = 0.3
GEN_TOP_P = 0.95
GEN_TOP_K = 50
GEN_REP_PENALTY = 1.10


# ═════════════════════════════════════════════════════════════════════
# PROMPT BATTERY
# ═════════════════════════════════════════════════════════════════════

EVAL_PROMPTS = [
    # --- Calibration (keyword-scored) ---
    {
        "prompt": "Maria has 3x as many apples as Joao. After Joao buys 12 more and Maria gives away a third of hers, they have the same amount. How many did Maria start with?",
        "check_words": ["36"], "anti_words": ["48", "24", "72"], "weight": 1.5,
    },
    {
        "prompt": "If all Bloops are Razzles, and all Razzles are Lazzles, but no Lazzles are Wazzles, what can we conclude about Bloops and Wazzles?",
        "check_words": ["no", "bloops", "wazzles"], "anti_words": [], "weight": 1.0,
    },
    {
        "prompt": "The element with atomic number 79 is gold. It shares a group with the element used in most electrical wiring. What are both elements?",
        "check_words": ["gold", "copper"], "anti_words": ["silver"], "weight": 1.0,
    },
    # --- Knowledge ---
    {
        "prompt": "What is the difference between a tensor, a vector, and a scalar in differential geometry? Give one concrete example of each.",
        "check_words": ["tensor", "vector", "scalar", "transform"], "anti_words": [], "weight": 1.0,
    },
    {
        "prompt": "What is the second law of thermodynamics, and how does it relate to the arrow of time?",
        "check_words": ["entropy", "increase", "disorder"], "anti_words": [], "weight": 1.0,
    },
    # --- Reasoning ---
    {
        "prompt": "A train travels from A to B at 80 km/h and returns at 120 km/h. What is the average speed for the round trip?",
        "check_words": ["96"], "anti_words": ["100"], "weight": 1.5,
    },
    {
        "prompt": "The philosopher who wrote the Critique of Pure Reason was born in the same city where he died. That city is now in which country?",
        "check_words": ["russia", "kaliningrad", "kant"], "anti_words": [], "weight": 1.0,
    },
    # --- Stress ---
    {
        "prompt": "Write a step-by-step proof that there are infinitely many primes.",
        "check_words": ["assume", "contradiction", "finite", "product"], "anti_words": [], "weight": 1.2,
    },
]

SYSTEM_PROMPT = "Answer directly with step-by-step reasoning. Be concise."


# ═════════════════════════════════════════════════════════════════════
# TELEMETRY STATE (shared with GSP bridge)
# ═════════════════════════════════════════════════════════════════════

TELEMETRY_STATE = {
    "curvature": -1.0,
    "entropy": 0.0,
    "active_fiber": "None",
    "constraint_score": 1.0,
    "history_k": [],
    "history_s": [],
}


# ═════════════════════════════════════════════════════════════════════
# MODEL LOADING (once per run)
# ═════════════════════════════════════════════════════════════════════

MODEL = {"llm": None, "tokenizer": None, "adapter": None}


def load_model():
    if MODEL["llm"] is not None:
        return

    print("[Benchmark] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config,
        device_map="auto", max_memory={0: "5GiB", "cpu": "30GiB"},
        trust_remote_code=True,
    )

    config = IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True,
        supported_modalities=["vision", "text"],
        use_delta_fiber=True, delta_mem_dim=64, delta_num_heads=4,
    )
    adapter = create_geometric_adapter(config).to(DEVICE)

    ms = ModelState()
    state = ms.load()
    weight_path = state["checkpoint"]["path"]
    if weight_path and os.path.exists(weight_path):
        print(f"[Benchmark] Loading adapter weights: {weight_path}", flush=True)
        adapter.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)

    _inject_adapter_hook(llm, adapter)

    llm.eval()
    MODEL["llm"] = llm
    MODEL["tokenizer"] = tokenizer
    MODEL["adapter"] = adapter

    vram_free = torch.cuda.mem_get_info()[0] / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"[Benchmark] Model loaded. VRAM free: {vram_free:.1f}GB", flush=True)


def _inject_adapter_hook(model, adapter):
    """Adapter hook at layer 12 with perturbation clamping + telemetry."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        layers = model.base_model.model.layers
    else:
        print("[Benchmark] ERROR: Cannot find transformer layers.")
        return

    target = layers[12]
    original_forward = target.forward

    def hook(hidden_states, *args, **kwargs):
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype

        h_in = h.to(DEVICE).to(torch.float32)
        with torch.no_grad():
            adapted_out, geo_state = adapter(h_in, pixel_values=None)

        # Telemetry
        try:
            probs = geo_state.fiber_sections.clamp(min=1e-8)
            S = float((-probs * probs.log()).sum(dim=-1).mean().item())
            TELEMETRY_STATE["entropy"] = S

            if hasattr(geo_state, "active_indices") and geo_state.active_indices is not None:
                idx = geo_state.active_indices.reshape(-1)[0].item()
                TELEMETRY_STATE["active_fiber"] = f"Bundle-{int(idx)}"

            if hasattr(adapter, "riemannian_geometry") and hasattr(geo_state, "base_coordinates"):
                try:
                    K = adapter.riemannian_geometry.estimate_sectional_curvature_stochastic(
                        geo_state.base_coordinates, num_samples=3
                    ).mean().item()
                    TELEMETRY_STATE["curvature"] = K
                except Exception:
                    pass
        except Exception:
            pass

        # Perturbation clamping (same as Neural Glass)
        adapted = adapted_out.to(orig_dtype)
        h_base = h.to(orig_dtype)
        delta = adapted - h_base
        base_norm = h_base.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        max_ratio = 0.10
        scale = torch.where(
            delta_norm > max_ratio * base_norm,
            max_ratio * base_norm / delta_norm,
            torch.ones_like(delta_norm),
        )
        adapted = h_base + delta * scale

        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted

    target.forward = hook
    print("[Benchmark] Adapter hook injected at layer 12.")


# ═════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════

def score_quality(response: str, prompt_info: dict) -> float:
    """Keyword + anti-keyword + degeneration scoring. Returns [0, 1]."""
    lower = response.lower()
    score = 0.0

    for w in prompt_info.get("check_words", []):
        if w.lower() in lower:
            score += 1.0 / max(1, len(prompt_info["check_words"]))

    for w in prompt_info.get("anti_words", []):
        if w.lower() in lower:
            score -= 0.3

    # Length penalties
    wc = len(response.split())
    if wc < 15:
        score -= 0.3
    elif wc > 400:
        score -= 0.1

    # Repetition detection (compression ratio)
    if len(response) > 50:
        compressed = len(zlib.compress(response.encode("utf-8")))
        ratio = compressed / max(1, len(response.encode("utf-8")))
        if ratio > 0.95:  # near-incompressible = heavy repetition
            score -= 0.4

    # Sentence uniqueness
    sentences = [s.strip().lower() for s in response.split(".") if s.strip()]
    if len(sentences) > 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.5:
            score -= 0.5

    return max(0.0, min(1.0, score))


def check_eos_natural(output_ids: torch.Tensor, input_len: int,
                      tokenizer, max_new_tokens: int) -> bool:
    """Check if generation stopped because of natural EOS (not hitting max_new_tokens)."""
    gen_len = output_ids.shape[-1] - input_len
    if gen_len >= max_new_tokens - 1:
        return False  # Hit the cap — no natural EOS

    # Check if last token is an EOS variant
    last_tok = output_ids[0, -1].item()
    eos_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        eos_ids.add(im_end)
    return last_tok in eos_ids


def detect_leakage(response: str) -> float:
    """Detect prompt leakage / persona injection. Returns penalty [0, 0.5]."""
    fragments = [
        "Answer directly with step-by-step",
        "You are Elizabeth",
        "izabeth, a highly intelligent",
        "<|im_start|>", "<|im_end|>",
        "user:", "assistant:",
    ]
    penalty = 0.0
    lower = response.lower()
    for f in fragments:
        if f.lower() in lower:
            penalty += 0.15
    return min(0.5, penalty)


# ═════════════════════════════════════════════════════════════════════
# GENERATION WITH GSP
# ═════════════════════════════════════════════════════════════════════

def generate_with_gsp(prompt: str, gsp, bridge, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a single response with GSP active. Returns (text, output_ids, input_len, telemetry_snapshot)."""
    llm = MODEL["llm"]
    tokenizer = MODEL["tokenizer"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(llm.device)
    input_len = inputs.input_ids.shape[-1]

    remaining = 32768 - input_len - 64
    safe_max = max(128, min(max_new_tokens, remaining))

    # EOS boost processor
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.eos_token_id:
        eos_ids.append(im_end_id)

    class _EOSBoost(LogitsProcessor):
        def __init__(self, eos_ids, prompt_len, min_tok=64, boost=0.015):
            self.eos_ids = eos_ids
            self.prompt_len = prompt_len
            self.min_tok = min_tok
            self.boost = boost
        def __call__(self, input_ids, scores):
            gen = input_ids.shape[-1] - self.prompt_len
            if gen > self.min_tok:
                b = self.boost * (gen - self.min_tok)
                for e in self.eos_ids:
                    if e is not None and 0 <= e < scores.shape[-1]:
                        scores[:, e] += b
            return scores

    # Poll GSP bridge (reads telemetry → PID gain)
    bridge.poll_and_update()

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=safe_max,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REP_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            logits_processor=LogitsProcessorList([_EOSBoost(eos_ids, input_len)]),
        )

    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    snap = {
        "S": TELEMETRY_STATE["entropy"],
        "K": TELEMETRY_STATE["curvature"],
        "fiber": TELEMETRY_STATE["active_fiber"],
    }

    del inputs
    torch.cuda.empty_cache()
    gc.collect()

    return text, outputs, input_len, snap


# ═════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION
# ═════════════════════════════════════════════════════════════════════

def evaluate_genome(genome: GSPGenome, n_prompts: int = 8,
                    verbose: bool = False) -> float:
    """
    Fitness function for CMA-ES.
    Attaches GSP with the given genome, runs n_prompts generations, scores.

    Returns scalar fitness (higher = better).
    """
    llm = MODEL["llm"]
    tokenizer = MODEL["tokenizer"]

    # Create GSP with this genome
    gsp, bridge = create_gsp_for_qwen7b(
        telemetry_dict=TELEMETRY_STATE, genome=genome
    )
    gsp.attach(llm, range(8, 21))

    # Select prompts (cycle through if n_prompts > len(EVAL_PROMPTS))
    prompts = []
    for i in range(n_prompts):
        prompts.append(EVAL_PROMPTS[i % len(EVAL_PROMPTS)])

    # Accumulators
    quality_scores = []
    eos_natural = []
    entropy_health = []
    leakage_penalties = []
    curvature_drifts = []
    baseline_K = TELEMETRY_STATE.get("curvature", -1.0)

    for i, pinfo in enumerate(prompts):
        try:
            gsp.reset_pid()
            text, output_ids, input_len, snap = generate_with_gsp(
                pinfo["prompt"], gsp, bridge
            )

            # 1. Quality
            q = score_quality(text, pinfo) * pinfo.get("weight", 1.0)
            quality_scores.append(q)

            # 2. EOS natural emission
            eos = check_eos_natural(output_ids, input_len, tokenizer, MAX_NEW_TOKENS)
            eos_natural.append(1.0 if eos else 0.0)

            # 3. Entropy health: S in [0.8, 1.4] band
            S = snap["S"]
            if 0.8 <= S <= 1.4:
                e_health = 1.0
            elif S < 0.5 or S > 2.0:
                e_health = 0.0
            else:
                e_health = 0.5
            entropy_health.append(e_health)

            # 4. Leakage penalty
            leak = detect_leakage(text)
            leakage_penalties.append(leak)

            # 5. Curvature drift from baseline
            K = snap["K"]
            drift = abs(K - baseline_K) / max(1.0, abs(baseline_K))
            curvature_drifts.append(min(1.0, drift))

            if verbose:
                gen_len = output_ids.shape[-1] - input_len
                print(f"  [{i+1}/{n_prompts}] Q={q:.2f} EOS={eos} "
                      f"S={S:.2f} K={K:.2f} leak={leak:.2f} "
                      f"len={gen_len} | {text[:60]}...")

        except Exception as e:
            print(f"  [{i+1}/{n_prompts}] ERROR: {e}")
            quality_scores.append(0.0)
            eos_natural.append(0.0)
            entropy_health.append(0.0)
            leakage_penalties.append(0.5)
            curvature_drifts.append(1.0)

    # Detach GSP hooks
    gsp.detach()

    # Compute weighted fitness
    n = max(1, len(quality_scores))
    f_quality   = sum(quality_scores) / n
    f_eos       = sum(eos_natural) / n
    f_entropy   = sum(entropy_health) / n
    f_leakage   = 1.0 - (sum(leakage_penalties) / n)  # higher = less leakage
    f_curvature = 1.0 - (sum(curvature_drifts) / n)    # higher = less drift

    # Weighted combination
    fitness = (
        0.30 * f_eos          # EOS natural emission is primary objective
      + 0.25 * f_quality      # Answer quality
      + 0.20 * f_entropy      # Entropy in healthy range
      + 0.15 * f_leakage      # No prompt leakage
      + 0.10 * f_curvature    # Curvature preserved
    )

    if verbose:
        print(f"  FITNESS={fitness:.4f} [eos={f_eos:.2f} qual={f_quality:.2f} "
              f"S={f_entropy:.2f} leak={f_leakage:.2f} K={f_curvature:.2f}]")

    return fitness


# ═════════════════════════════════════════════════════════════════════
# CMA-ES RUNNER
# ═════════════════════════════════════════════════════════════════════

def run_cmaes(n_generations: int = 30, popsize: int = 8,
              n_prompts: int = 8, sigma0: float = 0.1,
              output_dir: str = "gsp_benchmark_results"):
    """Run CMA-ES optimization of GSP genome."""

    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"gsp_cmaes_{ts}.json")

    print("=" * 60)
    print("GSP CMA-ES BENCHMARK")
    print(f"  Generations: {n_generations}")
    print(f"  Population:  {popsize}")
    print(f"  Prompts/eval: {n_prompts}")
    print(f"  Sigma0:      {sigma0}")
    print(f"  Output:      {log_file}")
    print("=" * 60)

    # Baseline evaluation (default genome)
    print("\n[Baseline] Evaluating default genome...", flush=True)
    default_genome = GSPGenome()
    baseline_fitness = evaluate_genome(default_genome, n_prompts=n_prompts, verbose=True)
    print(f"[Baseline] fitness={baseline_fitness:.4f}\n")

    # Evolution log
    history = {
        "config": {
            "n_generations": n_generations,
            "popsize": popsize,
            "n_prompts": n_prompts,
            "sigma0": sigma0,
        },
        "baseline": {
            "fitness": baseline_fitness,
            "genome": default_genome.to_vector().tolist(),
        },
        "generations": [],
        "best": None,
    }

    # CMA-ES or fallback
    try:
        import cma
        print("[CMA-ES] Using pycma", flush=True)
        x0 = default_genome.to_vector()
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {"maxiter": n_generations, "popsize": popsize, "verb_disp": 1},
        )

        gen = 0
        while not es.stop():
            gen += 1
            solutions = es.ask()

            # Evaluate population
            fitnesses = []
            for j, sol in enumerate(solutions):
                g = GSPGenome.from_vector(np.array(sol))
                # Clamp genome values to sane ranges
                g.K_P = max(0.001, min(0.5, g.K_P))
                g.K_I = max(0.0, min(0.2, g.K_I))
                g.K_D = max(0.0, min(0.1, g.K_D))
                g.S_target = max(0.5, min(2.5, g.S_target))
                g.tau_lock = max(0.1, min(0.95, g.tau_lock))
                g.mem0_semantic_alpha = max(0.0, min(1.0, g.mem0_semantic_alpha))
                g.lambda_max = max(0.001, min(0.5, g.lambda_max))
                g.layer_weights = [max(0.0, min(1.0, w)) for w in g.layer_weights]

                verbose = (j == 0)  # verbose for first member
                f = evaluate_genome(g, n_prompts=n_prompts, verbose=verbose)
                fitnesses.append(f)
                print(f"  gen={gen} ind={j+1}/{len(solutions)} F={f:.4f} "
                      f"S_t={g.S_target:.2f} lam_max={g.lambda_max:.3f}", flush=True)

            # CMA-ES minimizes, so negate
            es.tell(solutions, [-f for f in fitnesses])
            es.disp()

            # Log
            gen_data = {
                "generation": gen,
                "fitnesses": fitnesses,
                "best_fitness": max(fitnesses),
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "best_genome": GSPGenome.from_vector(
                    np.array(solutions[fitnesses.index(max(fitnesses))])
                ).to_vector().tolist(),
                "sigma": es.sigma,
            }
            history["generations"].append(gen_data)

            # Periodic save
            if gen % 5 == 0:
                with open(log_file, "w") as f:
                    json.dump(history, f, indent=2)
                print(f"  [Saved checkpoint at gen {gen}]", flush=True)

        # Final best
        best_vec = np.array(es.result.xbest)
        best_genome = GSPGenome.from_vector(best_vec)

    except ImportError:
        print("[CMA-ES] pycma not installed — using simple (1+λ)-ES", flush=True)
        best_genome = default_genome
        best_fitness = baseline_fitness
        sigma = sigma0

        for gen in range(1, n_generations + 1):
            x0 = best_genome.to_vector()
            candidates = []
            for _ in range(popsize):
                noise = np.random.randn(len(x0)) * sigma
                v = np.clip(x0 + noise, 0.0, 2.5)
                candidates.append(GSPGenome.from_vector(v))

            fitnesses = []
            for j, g in enumerate(candidates):
                f = evaluate_genome(g, n_prompts=n_prompts, verbose=(j == 0))
                fitnesses.append(f)
                print(f"  gen={gen} ind={j+1}/{popsize} F={f:.4f}", flush=True)

            # Update best
            best_idx = int(np.argmax(fitnesses))
            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_genome = candidates[best_idx]

            # 1/5 success rule
            n_better = sum(1 for f in fitnesses if f > best_fitness * 0.99)
            sigma = np.clip(
                sigma * (1.2 if n_better / popsize > 0.2 else 0.8),
                1e-5, 0.5,
            )

            gen_data = {
                "generation": gen,
                "fitnesses": fitnesses,
                "best_fitness": best_fitness,
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "best_genome": best_genome.to_vector().tolist(),
                "sigma": sigma,
            }
            history["generations"].append(gen_data)
            print(f"  gen={gen} best={best_fitness:.4f} sigma={sigma:.4f}", flush=True)

            if gen % 5 == 0:
                with open(log_file, "w") as f:
                    json.dump(history, f, indent=2)

    # Save final results
    print("\n" + "=" * 60)
    print("FINAL EVALUATION — best genome")
    print("=" * 60)
    final_fitness = evaluate_genome(best_genome, n_prompts=n_prompts, verbose=True)
    print(f"\nBaseline fitness:  {baseline_fitness:.4f}")
    print(f"Evolved fitness:   {final_fitness:.4f}")
    print(f"Improvement:       {final_fitness - baseline_fitness:+.4f}")

    history["best"] = {
        "fitness": final_fitness,
        "genome": best_genome.to_vector().tolist(),
        "genome_readable": {
            "K_P": best_genome.K_P,
            "K_I": best_genome.K_I,
            "K_D": best_genome.K_D,
            "S_target": best_genome.S_target,
            "tau_lock": best_genome.tau_lock,
            "mem0_semantic_alpha": best_genome.mem0_semantic_alpha,
            "lambda_max": best_genome.lambda_max,
            "layer_weights": best_genome.layer_weights,
        },
    }

    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved: {log_file}")

    # Also save best genome as standalone file
    genome_file = os.path.join(output_dir, f"best_genome_{ts}.json")
    with open(genome_file, "w") as f:
        json.dump(history["best"]["genome_readable"], f, indent=2)
    print(f"Best genome:  {genome_file}")

    return best_genome


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GSP CMA-ES Benchmark")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--output-dir", default="gsp_benchmark_results")
    args = parser.parse_args()

    load_model()
    best = run_cmaes(
        n_generations=args.generations,
        popsize=args.popsize,
        n_prompts=args.prompts,
        sigma0=args.sigma0,
        output_dir=args.output_dir,
    )
    print(f"\nBest genome vector: {best.to_vector().tolist()}")


if __name__ == "__main__":
    main()
