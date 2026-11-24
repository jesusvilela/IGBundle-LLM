"""Dimension-agnostic manifold telemetry for Neural Glass.

The adapter state is n-dimensional; any 2D view is only a projection for UI.
This module keeps the full latent tensor as the source of truth and derives
small scalar summaries for polling, logging, and dashboards.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


EPS = 1e-8


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            value = value.detach().float().mean().item()
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def _score_from_loss(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, math.exp(-abs(value)))))


def _tensor_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.float(), dim=-1)


def _poincare_radius(norm: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(norm.float(), min=0.0, max=1.0 - 1e-5)
    return 2.0 * torch.atanh(clipped)


def _poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    c = float(curvature) if curvature and curvature > 0 else 1.0
    x = x.float()
    y = y.float()
    x_norm_sq = torch.sum(x.pow(2), dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y.pow(2), dim=-1, keepdim=True)
    delta_sq = torch.sum((x - y).pow(2), dim=-1, keepdim=True)
    denom = ((1.0 - c * x_norm_sq) * (1.0 - c * y_norm_sq)).clamp_min(1e-6)
    arg = 1.0 + (2.0 * c * delta_sq / denom)
    return torch.acosh(arg.clamp_min(1.0 + 1e-6)) / math.sqrt(c)


def _project_point(point: torch.Tensor) -> list[float]:
    if point.ndim > 1:
        point = point.reshape(-1, point.shape[-1])[0]
    point = point.detach().float().cpu()
    dim = int(point.shape[-1]) if point.ndim else 0
    if dim <= 0:
        return [0.0, 0.0]
    x = _safe_float(point[0])
    y = _safe_float(point[1]) if dim > 1 else 0.0
    norm = math.sqrt(x * x + y * y)
    if norm >= 0.999:
        scale = 0.999 / max(norm, EPS)
        x *= scale
        y *= scale
    return [x, y]


def _adapter_dims(adapter: Any, coords: Optional[torch.Tensor], fibers: Optional[torch.Tensor]) -> Dict[str, Any]:
    cfg = getattr(adapter, "cfg", None)
    dims = {
        "hidden_dim": int(getattr(adapter, "H", getattr(cfg, "hidden_size", 0)) or 0),
        "latent_dim": int(getattr(adapter, "D", getattr(cfg, "latent_dim", 0)) or 0),
        "components": int(getattr(adapter, "P", getattr(cfg, "num_components", 0)) or 0),
        "categories": int(getattr(adapter, "K", getattr(cfg, "num_categories", 0)) or 0),
        "bottleneck_dim": int(getattr(adapter, "D_bot", getattr(cfg, "bottleneck_dim", 0)) or 0),
        "manifold_type": str(getattr(adapter, "manifold_type", getattr(cfg, "manifold_type", "unknown"))),
        "dynamics": bool(getattr(cfg, "use_dynamics", getattr(adapter, "use_dynamics", False))),
        "delta_fiber": bool(getattr(adapter, "use_delta_fiber", getattr(cfg, "use_delta_fiber", False))),
    }
    if coords is not None and coords.ndim >= 4:
        dims["components"] = int(coords.shape[-2])
        dims["latent_dim"] = int(coords.shape[-1])
    if fibers is not None and fibers.ndim >= 4:
        dims["categories"] = int(fibers.shape[-1])
    return dims


def _quality_vector(
    entropy_norm: float,
    sheaf_score: float,
    boundary_margin: float,
    adiabatic_score: float,
    active_coverage: float,
    hamiltonian_score: float,
    resonance_score: float,
    retrospection_score: float,
    constraint_alignment: float,
    meta_reflection: float,
) -> Dict[str, float]:
    identity = max(0.0, min(1.0, boundary_margin))
    otherness = max(0.0, min(1.0, entropy_norm))
    recognition = sheaf_score
    resonance = resonance_score
    recursion = max(0.0, min(1.0, (identity + recognition + otherness) / 3.0))
    godelian_remainder = max(0.0, min(1.0, 1.0 - recursion))
    ergodic_coverage = max(0.0, min(1.0, active_coverage))
    ergodetic_flow = max(0.0, min(1.0, 0.5 * adiabatic_score + 0.5 * ergodic_coverage))
    self_reflection = max(0.0, min(1.0, meta_reflection))
    coherence = max(0.0, min(1.0, 0.5 * recognition + 0.5 * constraint_alignment))
    plasticity = max(0.0, min(1.0, entropy_norm * boundary_margin))
    sparsity = max(0.0, min(1.0, 1.0 - active_coverage))
    stability = max(0.0, min(1.0, 0.5 * boundary_margin + 0.5 * adiabatic_score))
    reversibility = max(0.0, min(1.0, retrospection_score))
    exploratory_drive = max(0.0, min(1.0, entropy_norm * (1.0 - boundary_margin)))
    consolidation = max(0.0, min(1.0, identity * (1.0 - entropy_norm)))
    return {
        "identity": identity,
        "otherness": otherness,
        "mutual_recognition": recognition,
        "mutual_resonance": resonance,
        "recursion": recursion,
        "godelian_remainder": godelian_remainder,
        "adiabaticity": adiabatic_score,
        "hamiltonian_grounding": hamiltonian_score,
        "ergodic_coverage": ergodic_coverage,
        "ergodetic_flow": ergodetic_flow,
        "self_reflection": self_reflection,
        "coherence": coherence,
        "plasticity": plasticity,
        "sparsity": sparsity,
        "stability": stability,
        "reversibility": reversibility,
        "constraint_alignment": constraint_alignment,
        "exploratory_drive": exploratory_drive,
        "consolidation": consolidation,
    }


def summarize_manifold_stack(
    adapter: Any,
    geo_state: Any,
    hidden_state: Optional[torch.Tensor] = None,
    previous_geo: Optional[torch.Tensor] = None,
    previous_hidden: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    constraint_score: Optional[float] = None,
) -> Dict[str, Any]:
    coords = getattr(geo_state, "base_coordinates", None)
    fibers = getattr(geo_state, "fiber_sections", None)
    dims = _adapter_dims(adapter, coords, fibers)

    summary: Dict[str, Any] = {
        "dims": dims,
        "projection": [0.0, 0.0],
        "radius_mean": 0.0,
        "radius_max": 0.0,
        "hyperbolic_radius_mean": 0.0,
        "boundary_margin": 1.0,
        "geodesic_step": 0.0,
        "euclidean_step": 0.0,
        "holoportation_ratio": 1.0,
        "fiber_entropy_norm": 0.0,
        "sheaf_defect": 0.0,
        "bundle_spread": 0.0,
        "hamiltonian_energy": 0.0,
        "retrospection_loss": 0.0,
        "active_coverage": 0.0,
        "stack_depth": 0,
        "quality_vector": {},
    }

    current_geo = None
    if isinstance(coords, torch.Tensor) and coords.ndim >= 4:
        coord_norm = _tensor_norm(coords)
        radius_mean = _safe_float(coord_norm.mean())
        radius_max = _safe_float(coord_norm.max())
        summary["radius_mean"] = radius_mean
        summary["radius_max"] = radius_max
        summary["boundary_margin"] = max(0.0, 1.0 - radius_max)
        summary["hyperbolic_radius_mean"] = _safe_float(_poincare_radius(coord_norm).mean())
        summary["bundle_spread"] = _safe_float(coords.float().var(dim=-2, unbiased=False).mean())
        current_geo = coords[:, -1, 0, :].detach()
        summary["projection"] = _project_point(current_geo)

    entropy_norm = 0.0
    if isinstance(fibers, torch.Tensor):
        probs = fibers.float().clamp(min=EPS)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
        max_entropy = math.log(max(int(probs.shape[-1]), 2))
        entropy_norm = max(0.0, min(1.0, _safe_float(entropy) / max(max_entropy, EPS)))
        summary["fiber_entropy_norm"] = entropy_norm

    consensus_loss = getattr(geo_state, "consensus_loss", None)
    if consensus_loss is not None:
        sheaf_defect = _safe_float(consensus_loss)
    elif isinstance(fibers, torch.Tensor) and fibers.ndim >= 4 and fibers.shape[-2] > 1:
        sheaf_defect = _safe_float(fibers.float().var(dim=-2, unbiased=False).mean())
    else:
        sheaf_defect = 0.0
    summary["sheaf_defect"] = sheaf_defect
    sheaf_score = _score_from_loss(sheaf_defect)

    active_indices = getattr(geo_state, "active_indices", None)
    if isinstance(active_indices, torch.Tensor) and active_indices.numel() > 0:
        components = max(1, int(dims.get("components", 1)))
        unique = torch.unique(active_indices.detach().cpu()).numel()
        summary["active_coverage"] = min(1.0, float(unique) / float(components))

    meta_info = getattr(geo_state, "meta_info", None) or {}
    final_energy = meta_info.get("final_energy", meta_info.get("initial_energy", 0.0)) if isinstance(meta_info, dict) else 0.0
    summary["hamiltonian_energy"] = _safe_float(final_energy)
    hamiltonian_score = _score_from_loss(summary["hamiltonian_energy"])
    initial_energy = _safe_float(meta_info.get("initial_energy", final_energy), summary["hamiltonian_energy"]) if isinstance(meta_info, dict) else summary["hamiltonian_energy"]
    if isinstance(meta_info, dict) and meta_info.get("refined") and initial_energy > 0.0:
        meta_reflection = max(0.0, min(1.0, (initial_energy - summary["hamiltonian_energy"]) / (abs(initial_energy) + EPS)))
    else:
        meta_reflection = hamiltonian_score

    retrospection_loss = _safe_float(getattr(geo_state, "retrospection_loss", 0.0))
    summary["retrospection_loss"] = retrospection_loss
    retrospection_score = _score_from_loss(retrospection_loss)
    constraint_alignment = max(0.0, min(1.0, _safe_float(constraint_score, 1.0)))

    if current_geo is not None and isinstance(previous_geo, torch.Tensor):
        prev_geo = previous_geo.to(current_geo.device)
        if prev_geo.shape == current_geo.shape:
            geodesic = _poincare_distance(current_geo, prev_geo, curvature=curvature).mean()
            summary["geodesic_step"] = _safe_float(geodesic)

    if isinstance(hidden_state, torch.Tensor) and isinstance(previous_hidden, torch.Tensor):
        current_hidden = hidden_state[:, -1, :].detach()
        prev_hidden = previous_hidden.to(current_hidden.device)
        if prev_hidden.shape == current_hidden.shape:
            euclidean = torch.linalg.vector_norm(current_hidden.float() - prev_hidden.float(), dim=-1).mean()
            summary["euclidean_step"] = _safe_float(euclidean)

    euclidean_step = summary["euclidean_step"]
    geodesic_step = summary["geodesic_step"]
    if euclidean_step > 0.0:
        summary["holoportation_ratio"] = geodesic_step / (euclidean_step + EPS)
    adiabatic_score = _score_from_loss(max(0.0, summary["holoportation_ratio"] - 1.0))

    dims_for_depth = [
        dims.get("hidden_dim", 0),
        dims.get("latent_dim", 0),
        dims.get("components", 0),
        dims.get("categories", 0),
        dims.get("bottleneck_dim", 0),
    ]
    summary["stack_depth"] = sum(1 for value in dims_for_depth if int(value or 0) > 1)
    resonance_score = _score_from_loss(summary["bundle_spread"])
    summary["quality_vector"] = _quality_vector(
        entropy_norm=entropy_norm,
        sheaf_score=sheaf_score,
        boundary_margin=summary["boundary_margin"],
        adiabatic_score=adiabatic_score,
        active_coverage=summary["active_coverage"],
        hamiltonian_score=hamiltonian_score,
        resonance_score=resonance_score,
        retrospection_score=retrospection_score,
        constraint_alignment=constraint_alignment,
        meta_reflection=meta_reflection,
    )
    return summary


def format_stack_panel(summary: Dict[str, Any]) -> str:
    dims = summary.get("dims", {}) or {}
    qualities = summary.get("quality_vector", {}) or {}
    quality_order = [
        "identity",
        "otherness",
        "mutual_recognition",
        "mutual_resonance",
        "recursion",
        "godelian_remainder",
        "adiabaticity",
        "hamiltonian_grounding",
        "ergodic_coverage",
        "ergodetic_flow",
        "self_reflection",
        "coherence",
        "plasticity",
        "sparsity",
        "stability",
        "reversibility",
        "constraint_alignment",
        "exploratory_drive",
        "consolidation",
    ]
    quality_text = " ".join(
        f"{name[:4].upper()}={qualities.get(name, 0.0):.2f}" for name in quality_order
    )
    quality_lines = [quality_text[i:i + 94] for i in range(0, len(quality_text), 94)] or [""]
    return "\n".join([
        f"H={dims.get('hidden_dim', 0)} D={dims.get('latent_dim', 0)} P={dims.get('components', 0)} K={dims.get('categories', 0)} Bot={dims.get('bottleneck_dim', 0)}",
        f"M={dims.get('manifold_type', 'unknown')} dyn={int(bool(dims.get('dynamics')))} delta={int(bool(dims.get('delta_fiber')))} depth={summary.get('stack_depth', 0)}",
        f"r_mean={summary.get('radius_mean', 0.0):.3f} r_max={summary.get('radius_max', 0.0):.3f} margin={summary.get('boundary_margin', 0.0):.3f}",
        f"dM={summary.get('geodesic_step', 0.0):.4f} dE={summary.get('euclidean_step', 0.0):.4f} Lholo={summary.get('holoportation_ratio', 1.0):.3f}",
        f"S_norm={summary.get('fiber_entropy_norm', 0.0):.3f} sheaf={summary.get('sheaf_defect', 0.0):.5f} H={summary.get('hamiltonian_energy', 0.0):.5f} T={summary.get('retrospection_loss', 0.0):.5f}",
        *quality_lines,
    ])