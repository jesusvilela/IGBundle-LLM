"""Small prescribed-metric transport witness; no trained model or checkpoint."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from igbundle.geometry.poincare import PoincareBall


def lorentz_transport(x, y, v, c):
    """Independent hyperboloid embedding, transport, and inverse-chart derivative."""
    root = math.sqrt(c)

    def embed(point):
        norm2 = point.square().sum(dim=-1, keepdim=True)
        gap = 1 - c * norm2
        return torch.cat(((1 + c * norm2) / (root * gap), 2 * point / gap), dim=-1)

    def inner(a, b):
        return -(a[..., :1] * b[..., :1]) + (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)

    X, Y = embed(x), embed(y)
    gap = 1 - c * x.square().sum(dim=-1, keepdim=True)
    xv = (x * v).sum(dim=-1, keepdim=True)
    V = torch.cat((4 * root * xv / gap.square(),
                   2 * v / gap + 4 * c * x * xv / gap.square()), dim=-1)
    W = V + inner(Y, V) / (1 / c - inner(X, Y)) * (X + Y)
    denominator = root * Y[..., :1] + 1
    return W[..., 1:] / denominator - Y[..., 1:] * root * W[..., :1] / denominator.square()


def witness():
    torch.set_num_threads(1)
    ball = PoincareBall(2)
    origin = torch.tensor([0., 0.], dtype=torch.float64)
    a = torch.tensor([.3, 0.], dtype=torch.float64)
    b = torch.tensor([0., .4], dtype=torch.float64)
    initial = torch.tensor([1., 0.], dtype=torch.float64)
    actual, independent, scale_only = initial, initial, initial
    for start, end in ((origin, a), (a, b), (b, origin)):
        actual = ball.parallel_transport(start, end, actual)
        independent = lorentz_transport(start, end, independent, 1.)
        scale_only = scale_only * ball.conformal_factor(start) / ball.conformal_factor(end)
    angle = math.atan2(actual[1].item(), actual[0].item())
    expected = -2 * math.atan(.3 * .4)
    p = torch.tensor([.2, -.7], dtype=torch.float64)
    tangent = ball.parallel_transport(a, b, initial)
    covector = ball.cotangent_transport(a, b, p)
    pairing_error = abs(((tangent * covector).sum() - (initial * p).sum()).item())
    energy_before = p.square().sum() / (2 * ball.conformal_factor(a).square().squeeze())
    energy_after = covector.square().sum() / (2 * ball.conformal_factor(b).square().squeeze())
    wrong_p = ball.parallel_transport(a, b, p)
    wrong_energy = wrong_p.square().sum() / (2 * ball.conformal_factor(b).square().squeeze())
    result = {
        "scope": "prescribed K=-1 metric; no learned geometry, model, GGUF or integrator certification",
        "dtype": "float64",
        "triangle": [origin.tolist(), a.tolist(), b.tolist(), origin.tolist()],
        "initial_tangent": initial.tolist(),
        "returned_tangent": actual.tolist(),
        "signed_holonomy_radians": angle,
        "expected_negative_angle_defect_radians": expected,
        "angle_error": abs(angle - expected),
        "lorentz_reference_max_abs_error": (actual - independent).abs().max().item(),
        "scale_only_returned_tangent": scale_only.tolist(),
        "scale_only_signed_holonomy_radians": math.atan2(scale_only[1].item(), scale_only[0].item()),
        "cotangent_pairing_abs_error": pairing_error,
        "cotangent_kinetic_energy_abs_error": abs((energy_after - energy_before).item()),
        "tangent_applied_to_momentum_energy_ratio": (wrong_energy / energy_before).item(),
        "acceptance_absolute_tolerance": 1e-12,
    }
    for key in ("angle_error", "lorentz_reference_max_abs_error", "cotangent_pairing_abs_error",
                "cotangent_kinetic_energy_abs_error"):
        if not math.isfinite(result[key]) or result[key] > 1e-12:
            raise AssertionError(f"Transport witness failed: {key}={result[key]}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(witness(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")
