"""
Tier 3: Manifold Faithfulness Audit

Empirically validate the "100% manifold faithfulness" claim by computing
actual geometric metrics during evaluation — not from hardcoded telemetry.

Tests:
1. Curvature stability: K variance across eval batches
2. Fiber entropy: S not frozen at ln(K)
3. Fiber diversity: JSD > 0 (not degenerate uniform)
4. Parallel transport fidelity: PT is not identity
5. Geodesic triangle inequality: d(x,z) <= d(x,y) + d(y,z)
6. GENERIC energy conservation: drift < 1% per step
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("audit")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LN_K = np.log(16)  # ln(16) = 2.7726 — entropy at uniform over K=16


def load_adapter_and_model(checkpoint_dir: str, model_id: str):
    """Load geometric adapter from Config A checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.igbundle.modules.geometric_adapter import create_geometric_adapter
    from src.igbundle.core.config import IGBundleConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Must match the config used during training (train_odyssey_v3.py)
    config = IGBundleConfig(
        hidden_size=3584,
        num_components=8,
        latent_dim=64,
        num_categories=16,
        use_dynamics=True,
        supported_modalities=["vision", "text"],
    )
    adapter = create_geometric_adapter(config).to(DEVICE)

    # Load adapter weights (try both naming conventions)
    adapter_path = Path(checkpoint_dir) / "adapter.pt"
    if not adapter_path.exists():
        adapter_path = Path(checkpoint_dir) / "adapter_weights.pt"
    if adapter_path.exists():
        result = adapter.load_state_dict(
            torch.load(str(adapter_path), map_location=DEVICE), strict=False
        )
        if result.missing_keys:
            logger.warning(f"Missing keys (OK for audit): {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys (OK for audit): {len(result.unexpected_keys)} keys")
        logger.info(f"Loaded adapter from {adapter_path}")
    else:
        logger.warning(f"No adapter.pt or adapter_weights.pt in {checkpoint_dir} — using random init")

    adapter.eval()
    return model, adapter, tokenizer


def generate_hidden_states(model, tokenizer, num_batches: int = 100, seq_len: int = 128):
    """Generate random hidden states by running model on random token sequences.

    We feed random tokens through the model and extract hidden states at layer 12
    to get realistic activation distributions for the geometric audit.
    """
    layers = None
    for accessor in [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.base_model.model.model.layers,
    ]:
        try:
            layers = accessor(model)
            break
        except (AttributeError, IndexError):
            continue

    if layers is None:
        raise RuntimeError("Cannot find transformer layers")

    hidden_states_list = []
    capture = {}

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        capture["hidden"] = h.detach()

    hook = layers[12].register_forward_hook(capture_hook)

    # Use a simple repeated prompt to get consistent hidden states
    vocab_size = tokenizer.vocab_size
    rng = torch.Generator().manual_seed(42)

    for i in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=rng).to(DEVICE)
        with torch.no_grad():
            model(input_ids=input_ids)
        if "hidden" in capture:
            hidden_states_list.append(capture["hidden"].cpu())
        capture.clear()

        if (i + 1) % 20 == 0:
            logger.info(f"Generated hidden states: {i+1}/{num_batches}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hook.remove()
    return hidden_states_list


# ── Test 1: Curvature Stability ──────────────────────────────────

def test_curvature_stability(adapter, hidden_states_list):
    """K variance across eval batches. Pass: sigma(K) < 1.0"""
    K_values = []
    for h in hidden_states_list:
        h = h.to(DEVICE).float()
        with torch.no_grad():
            _, geo_state = adapter(h)
            K = adapter.riemannian_geometry.estimate_sectional_curvature_stochastic(
                geo_state.base_coordinates, num_samples=3
            )
            K_values.append(float(K.mean().item()))

    K_mean = float(np.mean(K_values))
    K_std = float(np.std(K_values))

    return {
        "K_mean": K_mean,
        "K_std": K_std,
        "pass": bool(K_std < 1.0),
    }


# ── Test 2: Fiber Section Entropy ────────────────────────────────

def test_fiber_entropy(adapter, hidden_states_list):
    """S not frozen at ln(K). Pass: |S - ln(K)| > 0.01"""
    S_values = []
    for h in hidden_states_list:
        h = h.to(DEVICE).float()
        with torch.no_grad():
            _, geo_state = adapter(h)
            p = geo_state.fiber_sections.clamp(min=1e-8)
            S = float(-(p * p.log()).sum(dim=-1).mean().item())
            S_values.append(S)

    S_mean = float(np.mean(S_values))
    S_std = float(np.std(S_values))
    frozen = bool(abs(S_mean - LN_K) < 0.01)

    return {
        "S_mean": S_mean,
        "S_std": S_std,
        "S_at_uniform": float(LN_K),
        "frozen": frozen,
        "pass": not frozen,
    }


# ── Test 3: Fiber Diversity (JSD) ────────────────────────────────

def test_fiber_diversity(adapter, hidden_states_list):
    """JSD(sections, uniform) > 0.01. Pass: routing is not degenerate."""
    JSD_values = []
    for h in hidden_states_list:
        h = h.to(DEVICE).float()
        with torch.no_grad():
            _, geo_state = adapter(h)
            sections = geo_state.fiber_sections
            K_dim = sections.shape[-1]
            uniform = torch.ones_like(sections) / K_dim
            m = 0.5 * (sections + uniform)
            kl_p_m = (sections * (sections.clamp(min=1e-8).log() - m.clamp(min=1e-8).log())).sum(-1)
            kl_u_m = (uniform * (uniform.log() - m.clamp(min=1e-8).log())).sum(-1)
            jsd = float((0.5 * (kl_p_m + kl_u_m)).mean().item())
            JSD_values.append(jsd)

    jsd_mean = float(np.mean(JSD_values))
    return {
        "jsd_mean": jsd_mean,
        "pass": bool(jsd_mean > 0.01),
    }


# ── Test 4: Parallel Transport Fidelity ──────────────────────────

def test_parallel_transport(adapter, hidden_states_list):
    """PT is not identity: ||PT(v) - v||/||v|| for random tangent vectors.
    Pass: ratio < 0.5 (transport actually moves vectors, but not too much).
    Also check: ratio > 0.001 (transport is not trivially identity)."""
    ratios = []
    for h in hidden_states_list[:50]:  # Subset for speed
        h = h.to(DEVICE).float()
        with torch.no_grad():
            _, geo_state = adapter(h)
            coords = geo_state.base_coordinates  # (B, T, P, D)
            metric = geo_state.metric

            # Call the actual parallel transport
            transported = adapter._parallel_transport_update(coords, metric)

            # Displacement ratio: how much did PT change the coordinates?
            diff = (transported - coords).norm(dim=-1)
            base_norm = coords.norm(dim=-1).clamp(min=1e-8)
            ratio = float((diff / base_norm).mean().item())
            ratios.append(ratio)

    mean_ratio = float(np.mean(ratios))
    return {
        "mean_displacement_ratio": mean_ratio,
        "is_identity": bool(mean_ratio < 0.001),
        "pass": bool(0.001 < mean_ratio < 0.5),
    }


# ── Test 5: Geodesic Triangle Inequality ─────────────────────────

def test_geodesic_triangle(adapter, hidden_states_list):
    """d(x,z) <= d(x,y) + d(y,z) for random triples. Pass: 100% satisfaction."""
    violations = 0
    total_triples = 0

    for h in hidden_states_list[:30]:  # Subset for speed
        h = h.to(DEVICE).float()
        with torch.no_grad():
            _, geo_state = adapter(h)
            coords = geo_state.base_coordinates  # (B, T, P, D)

            # Flatten to (N, D) for pairwise distances
            flat = coords.reshape(-1, coords.shape[-1])
            N = min(flat.shape[0], 100)
            flat = flat[:N]

            # Poincare ball distance: d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
            norms_sq = (flat ** 2).sum(-1)  # (N,)
            # Clamp norms to be inside the ball
            norms_sq = norms_sq.clamp(max=0.99)

            # Sample random triples
            n_triples = min(100, N * (N - 1) * (N - 2) // 6)
            rng = torch.Generator().manual_seed(42)
            for _ in range(n_triples):
                i, j, k = torch.randint(0, N, (3,), generator=rng).tolist()
                if i == j or j == k or i == k:
                    continue

                def poincare_dist(a, b):
                    diff_sq = ((a - b) ** 2).sum()
                    na_sq = (a ** 2).sum().clamp(max=0.99)
                    nb_sq = (b ** 2).sum().clamp(max=0.99)
                    denom = (1 - na_sq) * (1 - nb_sq)
                    arg = 1 + 2 * diff_sq / denom.clamp(min=1e-8)
                    return torch.acosh(arg.clamp(min=1.0 + 1e-6))

                d_ij = poincare_dist(flat[i], flat[j])
                d_jk = poincare_dist(flat[j], flat[k])
                d_ik = poincare_dist(flat[i], flat[k])

                total_triples += 1
                if d_ik > d_ij + d_jk + 1e-4:  # Small tolerance for numerics
                    violations += 1

    satisfaction = 1.0 - (violations / max(total_triples, 1))
    return {
        "violations": violations,
        "total_triples": total_triples,
        "satisfaction_rate": float(satisfaction),
        "pass": bool(violations == 0),
    }


# ── Test 6: GENERIC Energy Conservation ──────────────────────────

def test_energy_conservation(adapter, hidden_states_list):
    """Energy drift < 1% per integration step. Pass: max_drift_pct < 0.01"""
    drifts = []

    for h in hidden_states_list[:30]:
        h = h.to(DEVICE).float()
        with torch.no_grad():
            _, geo_state = adapter(h)
            coords = geo_state.base_coordinates
            sections = geo_state.fiber_sections
            metric = geo_state.metric

            # Energy before: kinetic (0 momentum) + potential (norm-based proxy)
            E_before = float((coords ** 2).sum(-1).mean().item())

            # Run symplectic integration
            if hasattr(adapter, '_symplectic_integrate'):
                q_new, s_new = adapter._symplectic_integrate(
                    coords, sections, metric
                )
                E_after = float((q_new ** 2).sum(-1).mean().item())

                if abs(E_before) > 1e-8:
                    drift = abs(E_after - E_before) / abs(E_before)
                    drifts.append(drift)

    if not drifts:
        return {"max_drift_pct": 1.0, "mean_drift_pct": 1.0, "pass": False,
                "note": "symplectic_integrate not available or returned unchanged"}

    max_drift = float(np.max(drifts))
    mean_drift = float(np.mean(drifts))
    return {
        "max_drift_pct": max_drift,
        "mean_drift_pct": mean_drift,
        "pass": bool(max_drift < 0.01),
    }


# ── Main ──────────────────────────────────────────────────────────

def run_audit(checkpoint_dir: str, model_id: str, output_path: str, num_batches: int = 100):
    logger.info("=" * 60)
    logger.info("MANIFOLD FAITHFULNESS AUDIT — Tier 3")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info("=" * 60)

    model, adapter, tokenizer = load_adapter_and_model(checkpoint_dir, model_id)

    logger.info(f"Generating {num_batches} hidden state batches...")
    hidden_states = generate_hidden_states(model, tokenizer, num_batches=num_batches)
    logger.info(f"Got {len(hidden_states)} batches")

    results = {}

    logger.info("\n[1/6] Curvature stability...")
    results["curvature_stability"] = test_curvature_stability(adapter, hidden_states)
    logger.info(f"  K_mean={results['curvature_stability']['K_mean']:.4f}, "
                f"K_std={results['curvature_stability']['K_std']:.4f}, "
                f"pass={results['curvature_stability']['pass']}")

    logger.info("[2/6] Fiber entropy...")
    results["fiber_entropy"] = test_fiber_entropy(adapter, hidden_states)
    logger.info(f"  S_mean={results['fiber_entropy']['S_mean']:.4f}, "
                f"frozen={results['fiber_entropy']['frozen']}, "
                f"pass={results['fiber_entropy']['pass']}")

    logger.info("[3/6] Fiber diversity (JSD)...")
    results["fiber_diversity"] = test_fiber_diversity(adapter, hidden_states)
    logger.info(f"  JSD_mean={results['fiber_diversity']['jsd_mean']:.6f}, "
                f"pass={results['fiber_diversity']['pass']}")

    logger.info("[4/6] Parallel transport fidelity...")
    results["parallel_transport"] = test_parallel_transport(adapter, hidden_states)
    logger.info(f"  ratio={results['parallel_transport']['mean_displacement_ratio']:.6f}, "
                f"is_identity={results['parallel_transport']['is_identity']}, "
                f"pass={results['parallel_transport']['pass']}")

    logger.info("[5/6] Geodesic triangle inequality...")
    results["geodesic_triangle"] = test_geodesic_triangle(adapter, hidden_states)
    logger.info(f"  violations={results['geodesic_triangle']['violations']}/"
                f"{results['geodesic_triangle']['total_triples']}, "
                f"pass={results['geodesic_triangle']['pass']}")

    logger.info("[6/6] GENERIC energy conservation...")
    results["generic_energy"] = test_energy_conservation(adapter, hidden_states)
    logger.info(f"  max_drift={results['generic_energy']['max_drift_pct']:.6f}, "
                f"pass={results['generic_energy']['pass']}")

    # Overall verdict
    all_pass = all(r["pass"] for r in results.values())
    n_pass = sum(1 for r in results.values() if r["pass"])

    output = {
        "config": "A",
        "checkpoint_dir": checkpoint_dir,
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
        "curvature_stability": results["curvature_stability"],
        "fiber_entropy": results["fiber_entropy"],
        "fiber_diversity": results["fiber_diversity"],
        "parallel_transport": results["parallel_transport"],
        "geodesic_triangle": results["geodesic_triangle"],
        "generic_energy": results["generic_energy"],
        "overall_faithfulness": all_pass,
        "tests_passed": f"{n_pass}/6",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT: {'FAITHFUL' if all_pass else 'NOT FAITHFUL'} ({n_pass}/6 tests passed)")
    logger.info(f"Results: {output_path}")
    logger.info(f"{'='*60}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifold Faithfulness Audit (Tier 3)")
    parser.add_argument("--checkpoint", required=True, help="Config A checkpoint directory")
    parser.add_argument("--model-id", default="h:/LLM-MANIFOLD/igbundle_qwen7b_cp600")
    parser.add_argument("--output", default="experiments/falsification/results/A_faithfulness_audit.json")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of eval batches")
    args = parser.parse_args()

    run_audit(args.checkpoint, args.model_id, args.output, args.num_batches)
