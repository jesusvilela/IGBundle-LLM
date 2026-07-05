"""kappa_invariance_test.py — Reproducible verification of §2.6 of
draft_paper_falsification.md: the legacy curvature telemetry is invariant
w.r.t. trained weights (an architecture constant), and the 2026-07 remediation
(learnable conformal factor) makes curvature weight-sensitive.

CPU-only, seconds to run. Requires only the cp3000 adapter_weights.pt
(58,705,739 bytes, SHA256 2004373636B049FC03771EE087FF1E4053D8003C18C8F3FA94668D598145DB14).

Usage:
    python scripts/kappa_invariance_test.py \
        [--checkpoint ../igbundle_phase8_training/checkpoint-3000/adapter_weights.pt]

Test A (legacy replica): runs the pre-audit estimator logic
    kappa = -0.5 * Lap(log det g) / D,   g(x) = lambda(x) * (LL^T + eps*I),
    lambda(x) = 1 + 0.5 * tanh||x||   (hardcoded)
under four metric_chol settings at IDENTICAL positions and finite-difference
directions:
    1. trained   (cp3000 checkpoint)
    2. identity  (untrained init, no noise)
    3. analytic  (chol-free closed form: -(D/2) * d^2/dx_k^2 log lambda)
    4. random    (control; expected to deviate via log-det clamp saturation)
Expected: (1) ~= (2) ~= (3) to <0.1% — proving the published K values contained
no trained parameter.

Test B (remediation): the corrected estimator in the current codebase responds
to the (now optionally trainable) conformal amplitude. Skipped gracefully if
the igbundle package cannot be imported.

Test C (radius sweep): reproduces the magnitude gradient that explains the
published values (large |kappa| near origin -> small at the r=0.95 projection
boundary, where -5.63/-5.72 were read).
"""
import argparse
import math
import os
import sys

import torch

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "igbundle_phase8_training", "checkpoint-3000", "adapter_weights.pt",
)
D = 64   # latent_dim of cp3000 (train_refined_hf.py)
P = 8    # num_components of cp3000
EPS_FD = 1e-3
SEED = 1234


# ----------------------------------------------------------------------------
# Legacy estimator replica (verbatim logic of the pre-audit
# RiemannianGeometry.get_metric + estimate_sectional_curvature_stochastic)
# ----------------------------------------------------------------------------

def legacy_metric(positions: torch.Tensor, metric_chol: torch.Tensor) -> torch.Tensor:
    """g(x) = lambda(x) * (tril(clamp(L)) tril(clamp(L))^T + 1e-5 I)."""
    B, T, Pn, Dn = positions.shape
    L = torch.tril(metric_chol.to(positions.dtype))
    L = torch.clamp(L, min=-5.0, max=5.0)
    metric = torch.matmul(L, L.transpose(-1, -2))
    eye = torch.eye(Dn, dtype=positions.dtype).expand_as(metric)
    metric = metric + 1e-5 * eye
    metric = metric.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    norm_x = torch.norm(positions, dim=-1, keepdim=True)
    conformal = (1.0 + 0.5 * torch.tanh(norm_x)).unsqueeze(-1)
    return metric * conformal


def safe_log_det(metric: torch.Tensor) -> torch.Tensor:
    _, logabsdet = torch.slogdet(metric)
    return logabsdet.clamp(-50.0, 50.0)


def legacy_kappa(positions: torch.Tensor, metric_chol: torch.Tensor,
                 directions: list) -> torch.Tensor:
    """kappa = -0.5 * Lap(log det g) / D with the legacy D-scaling."""
    base = safe_log_det(legacy_metric(positions, metric_chol))
    lap = torch.zeros_like(base)
    for k in directions:
        plus = positions.clone(); plus[..., k] += EPS_FD
        minus = positions.clone(); minus[..., k] -= EPS_FD
        f_p = safe_log_det(legacy_metric(plus, metric_chol))
        f_m = safe_log_det(legacy_metric(minus, metric_chol))
        lap = lap + (f_p + f_m - 2.0 * base) / (EPS_FD * EPS_FD)
    lap = lap * (D / len(directions))
    return -0.5 * lap / D


def analytic_kappa(positions: torch.Tensor, directions: list) -> torch.Tensor:
    """Chol-free closed form: -(D/2) * mean_k d^2/dx_k^2 log(1 + 0.5 tanh||x||).

    Contains NO trained parameter. Agreement with legacy_kappa(trained) is the
    proof of weight-invariance: log det g = D log lambda + const(x), and the
    constant cancels in finite differences.
    """
    def log_lam(x):
        return torch.log(1.0 + 0.5 * torch.tanh(torch.norm(x, dim=-1)))

    base = log_lam(positions)
    lap = torch.zeros_like(base)
    for k in directions:
        plus = positions.clone(); plus[..., k] += EPS_FD
        minus = positions.clone(); minus[..., k] -= EPS_FD
        lap = lap + (log_lam(plus) + log_lam(minus) - 2.0 * base) / (EPS_FD * EPS_FD)
    lap = lap * (D / len(directions))
    return -0.5 * D * lap / D  # = -(D/2) * mean second derivative, legacy scaling


def find_metric_chol(state_dict: dict) -> torch.Tensor:
    candidates = [k for k in state_dict if k.endswith("metric_chol")]
    if not candidates:
        raise KeyError(
            f"No 'metric_chol' key in checkpoint. Keys sample: {list(state_dict)[:10]}")
    chol = state_dict[candidates[0]].double().cpu()
    print(f"  metric_chol key: {candidates[0]}  shape: {tuple(chol.shape)}")
    return chol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    torch.manual_seed(SEED)

    # Shared positions and FD directions — identical for every metric setting.
    # Positions must be BALL-INTERIOR (||x|| < 0.95, as produced by
    # _project_to_ball in the adapter): with raw randn in D=64 the norm is
    # ~sqrt(D)*sigma, deep in tanh saturation, and the finite differences of a
    # float32 slogdet reduce to cancellation noise. float64 + realistic norms
    # are required for a clean invariance readout.
    raw = torch.randn(1, 4, P, D, dtype=torch.float64)
    target_norms = torch.rand(1, 4, P, 1, dtype=torch.float64) * 0.75 + 0.15  # in [0.15, 0.9]
    positions = raw / raw.norm(dim=-1, keepdim=True) * target_norms
    directions = [int(i) for i in torch.randint(0, D, (16,))]

    print("=" * 72)
    print("TEST A — legacy estimator invariance w.r.t. metric_chol")
    print("=" * 72)

    results = {}

    ckpt_path = os.path.abspath(args.checkpoint)
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        # weights_only=True: never unpickle arbitrary objects from a checkpoint.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            state = state.state_dict()
        trained_chol = find_metric_chol(state)
        results["trained (cp3000)"] = legacy_kappa(positions, trained_chol, directions).mean().item()
        # Report how far the trained metric sits from identity (audit: ~= I)
        ident = torch.eye(D, dtype=torch.float64).unsqueeze(0).repeat(trained_chol.shape[0], 1, 1)
        rel = (trained_chol - ident).norm() / ident.norm()
        print(f"  ||trained_chol - I|| / ||I|| = {rel.item():.4f}")
    else:
        print(f"Checkpoint not found at {ckpt_path} — skipping trained row.")

    identity_chol = torch.eye(D, dtype=torch.float64).unsqueeze(0).repeat(P, 1, 1)
    results["identity"] = legacy_kappa(positions, identity_chol, directions).mean().item()

    results["analytic (chol-free)"] = analytic_kappa(positions, directions).mean().item()

    random_chol = torch.randn(P, D, D, dtype=torch.float64)
    results["random (control)"] = legacy_kappa(positions, random_chol, directions).mean().item()

    width = max(len(k) for k in results)
    for name, val in results.items():
        print(f"  {name:<{width}} : mean kappa = {val: .3f}")

    if "trained (cp3000)" in results:
        rel_dev = abs(results["trained (cp3000)"] - results["identity"]) / abs(results["identity"])
        print(f"\n  trained vs identity relative deviation: {rel_dev:.2e}")
        verdict = "WEIGHT-INVARIANT (architecture constant)" if rel_dev < 1e-3 \
            else "weight-sensitive (unexpected — investigate)"
        print(f"  VERDICT: {verdict}")

    print()
    print("=" * 72)
    print("TEST C — radius sweep (explains published magnitudes)")
    print("=" * 72)
    for r in (0.1, 0.3, 0.5, 0.7, 0.9):
        pos_r = torch.zeros(1, 1, P, D, dtype=torch.float64)
        pos_r[..., 0] = r
        k_r = analytic_kappa(pos_r, list(range(0, D, 8))).mean().item()
        print(f"  r = {r:.1f} : kappa = {k_r: .3f}")
    print("  (the published -5.63/-5.72 are this curve read near the r=0.95")
    print("   projection boundary; the spread across published values is the")
    print("   coordinate-norm distribution shifting between evals)")

    print()
    print("=" * 72)
    print("TEST B — remediation: corrected estimator responds to conformal scale")
    print("=" * 72)
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from igbundle.core.config import IGBundleConfig
        from igbundle.geometry.riemannian import RiemannianGeometry
    except Exception as exc:  # pragma: no cover
        print(f"  SKIPPED (igbundle import failed: {exc})")
        return

    cfg = IGBundleConfig(latent_dim=D, num_components=P, num_categories=16,
                         manifold_type="riemannian", learnable_conformal=True)
    geo = RiemannianGeometry(cfg)
    pos32 = positions.float()
    torch.manual_seed(SEED)
    k_a = geo.estimate_sectional_curvature_stochastic(pos32, num_samples=8).mean().item()
    with torch.no_grad():
        geo.conformal_scale.fill_(0.05)
    torch.manual_seed(SEED)
    k_b = geo.estimate_sectional_curvature_stochastic(pos32, num_samples=8).mean().item()
    print(f"  conformal_scale=0.50 : kappa = {k_a: .6f}")
    print(f"  conformal_scale=0.05 : kappa = {k_b: .6f}")
    ok = abs(k_a - k_b) > 1e-8
    print(f"  VERDICT: {'RESPONSIVE — regularizer can bind' if ok else 'unresponsive (unexpected)'}")


if __name__ == "__main__":
    main()
