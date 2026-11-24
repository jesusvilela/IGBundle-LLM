"""Runtime EPIC coverage telemetry for IGBundle.

This module does not claim that an EPIC exists because a filename says so.
It reads live adapter/state objects and reports which geometric surfaces are
present, active, or absent at inference time.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch


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


def _status(active: bool, present: bool = True) -> str:
    if active:
        return "active"
    if present:
        return "present"
    return "absent"


def _has_nonempty_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _active_attr(obj: Any, names: Iterable[str]) -> bool:
    for name in names:
        if getattr(obj, name, None) is not None:
            return True
    return False


def _ood_state(n_stack: Dict[str, Any], telemetry_state: Dict[str, Any]) -> Dict[str, Any]:
    radius = _safe_float(n_stack.get("radius_max"), 0.0)
    holo = _safe_float(n_stack.get("holoportation_ratio"), 1.0)
    sheaf = _safe_float(n_stack.get("sheaf_defect"), 0.0)
    constraint = _safe_float(telemetry_state.get("constraint_score"), 1.0)
    curvature_status = str(telemetry_state.get("curvature_status", "unknown"))

    radius_pressure = max(0.0, min(1.0, radius / 0.95))
    holo_pressure = max(0.0, min(1.0, (holo - 1.0) / 7.0))
    sheaf_pressure = max(0.0, min(1.0, sheaf))
    constraint_pressure = max(0.0, min(1.0, 1.0 - constraint))
    estimator_pressure = 0.2 if curvature_status.startswith("failed") else 0.0
    score = max(radius_pressure, holo_pressure, sheaf_pressure, constraint_pressure, estimator_pressure)
    active = radius > 0.90 or holo > 8.0 or sheaf > 0.75 or constraint < 0.15
    reasons = []
    if radius > 0.90:
        reasons.append("boundary")
    if holo > 8.0:
        reasons.append("holoportation")
    if sheaf > 0.75:
        reasons.append("sheaf")
    if constraint < 0.15:
        reasons.append("constraint")
    if curvature_status.startswith("failed"):
        reasons.append("curvature-estimator")

    return {
        "active": bool(active),
        "score": float(score),
        "radius": radius,
        "holoportation_ratio": holo,
        "sheaf_defect": sheaf,
        "constraint_score": constraint,
        "reasons": reasons,
    }


def summarize_epic_stack(
    adapter: Any,
    geo_state: Any,
    telemetry_state: Optional[Dict[str, Any]] = None,
    gsp: Any = None,
    memory: Any = None,
) -> Dict[str, Any]:
    telemetry_state = telemetry_state or {}
    n_stack = telemetry_state.get("n_stack", {}) or {}
    meta_info = getattr(geo_state, "meta_info", None) or {}
    op_logits = getattr(geo_state, "op_logits", None)
    consensus_loss = getattr(geo_state, "consensus_loss", None)
    active_indices = getattr(geo_state, "active_indices", None)
    fiber_executor = getattr(adapter, "fiber_executor", None)
    provenance = telemetry_state.get("adapter_provenance", {}) or {}

    items = [
        {"id": "1", "name": "Hamiltonian", "status": _status(bool(getattr(adapter, "use_dynamics", False)), _active_attr(adapter, ["vf", "dynamics", "integrator"]))},
        {"id": "2", "name": "GeodesicAttention", "status": _status(bool(getattr(adapter, "use_geodesic_attn", False)), hasattr(adapter, "geodesic_attn"))},
        {"id": "3", "name": "TaskCompiler", "status": _status(_has_nonempty_tensor(op_logits), hasattr(adapter, "compiler"))},
        {"id": "5", "name": "MultimodalVision", "status": _status(getattr(adapter, "vision_projector", None) is not None, hasattr(adapter, "vision_projector"))},
        {"id": "6", "name": "PhaseMemory", "status": _status(_active_attr(adapter, ["phase_memory"]), _active_attr(adapter, ["phase_memory"]))},
        {"id": "7", "name": "SheafConsensus", "status": _status(consensus_loss is not None, hasattr(adapter, "consensus"))},
        {"id": "15", "name": "CheckpointLineage", "status": _status(bool(provenance.get("checkpoint_path") or provenance.get("mode")), True)},
        {"id": "17b", "name": "DeltaFiber", "status": _status(bool(getattr(adapter, "use_delta_fiber", False)), hasattr(adapter, "delta_fiber_update"))},
        {"id": "31", "name": "FiberLatents", "status": _status(_active_attr(adapter, ["fiber_store", "fiber_executor"]), _active_attr(adapter, ["fiber_store", "fiber_executor"]))},
        {"id": "35", "name": "SemanticPotential", "status": _status(_active_attr(adapter, ["potential_net"]), _active_attr(adapter, ["potential_net"]))},
        {"id": "36", "name": "MetaCognition", "status": _status(bool(meta_info), getattr(adapter, "meta_loop", None) is not None)},
        {"id": "41", "name": "ResonantFiber", "status": _status(getattr(fiber_executor, "resonator_state", None) is not None, fiber_executor is not None)},
        {"id": "42", "name": "OODDetector", "status": "active" if _ood_state(n_stack, telemetry_state)["active"] else "present"},
        {"id": "43", "name": "HolographicView", "status": _status(bool(n_stack.get("projection")), bool(n_stack))},
        {"id": "52", "name": "EvolutionaryMemory", "status": _status(memory is not None, True)},
    ]

    active_count = sum(1 for item in items if item["status"] == "active")
    present_count = sum(1 for item in items if item["status"] in {"active", "present"})
    ood = _ood_state(n_stack, telemetry_state)
    return {
        "items": items,
        "active_count": active_count,
        "present_count": present_count,
        "total_count": len(items),
        "coverage": active_count / max(len(items), 1),
        "presence": present_count / max(len(items), 1),
        "curvature_status": telemetry_state.get("curvature_status", "unknown"),
        "ood": ood,
    }


def format_epic_panel(summary: Dict[str, Any]) -> str:
    items = summary.get("items", []) or []
    ood = summary.get("ood", {}) or {}
    active = int(summary.get("active_count", 0))
    present = int(summary.get("present_count", 0))
    total = int(summary.get("total_count", len(items)))
    curvature_status = str(summary.get("curvature_status", "unknown"))[:24]
    header = f"active={active}/{total} present={present}/{total} OOD={ood.get('score', 0.0):.2f} K={curvature_status}"
    if ood.get("active"):
        reasons = ",".join(ood.get("reasons", []) or ["unknown"])
        header = f"{header} ALERT={reasons}"

    tokens = []
    for item in items:
        mark = "on" if item["status"] == "active" else "rd" if item["status"] == "present" else "--"
        tokens.append(f"E{item['id']}:{mark}")

    rows = [header]
    for start in range(0, len(tokens), 5):
        rows.append(" ".join(tokens[start:start + 5]))
    return "\n".join(rows)