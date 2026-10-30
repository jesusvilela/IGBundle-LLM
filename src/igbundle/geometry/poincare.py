"""Opt-in geodesic transport for the prescribed Poincare metric, K = -c.

This reference is not a drop-in replacement for archived dynamics and is not
wired into either adapter. In particular, momenta are covectors, not tangents.
"""

import math

import torch
from torch import nn


class PoincareBall(nn.Module):
    """Transport on g_x = [2 / (1 - c ||x||^2)]^2 I, for fixed c > 0."""

    def __init__(self, dim: int, c: float = 1.0):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 2:
            raise ValueError("Transport reference requires integer dim >= 2")
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
            raise ValueError("c must be a finite positive real number")
        self.dim = dim
        self.c = float(c)

    def _prepare(self, *values):
        for value in values:
            if (not isinstance(value, torch.Tensor) or value.ndim < 1
                    or value.shape[-1] != self.dim or not value.is_floating_point()
                    or value.layout != torch.strided or value.is_meta):
                raise ValueError("Expected dense floating tensors with final dimension dim")
            if value.device != values[0].device or value.dtype != values[0].dtype:
                raise ValueError("Transport inputs must share device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError("Transport inputs must be finite")
        work = tuple(value.float() if value.dtype in (torch.float16, torch.bfloat16)
                     else value for value in values)
        try:
            return torch.broadcast_tensors(*work)
        except RuntimeError as exc:
            raise ValueError("Transport batch dimensions must broadcast") from exc

    def _ball_coordinates(self, x):
        unit = math.sqrt(self.c) * x
        gap = 1 - unit.square().sum(dim=-1, keepdim=True)
        if not torch.isfinite(gap).all() or (gap <= 0).any():
            raise ValueError("Points must lie strictly inside the curvature-scaled ball")
        return unit, gap

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Return lambda_x; half/bfloat16 inputs are evaluated in float32."""
        (x,) = self._prepare(x)
        _, gap = self._ball_coordinates(x)
        return 2 / gap

    @staticmethod
    def _gyration(u, v, w):
        """Linear gyr[u,v] w, with u,v in the unit-curvature ball."""
        u2 = u.square().sum(dim=-1, keepdim=True)
        v2 = v.square().sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        denominator = 1 + 2 * uv + u2 * v2
        if (denominator <= 0).any():
            raise ValueError("Gyration denominator lost positivity at this precision")
        a = -uw * v2 + vw + 2 * uv * vw
        b = -vw * u2 - uw
        return w + 2 * (a * u + b * v) / denominator

    def _transport(self, x, y, value, *, covector):
        x, y, value = self._prepare(x, y, value)
        ux, gap_x = self._ball_coordinates(x)
        uy, gap_y = self._ball_coordinates(y)
        rotation = self._gyration(uy, -ux, value)
        scale = gap_x / gap_y if covector else gap_y / gap_x
        result = scale * rotation
        if not torch.isfinite(result).all():
            raise ValueError("Transport overflow at this precision")
        return result

    def parallel_transport(self, x, y, vector):
        """Transport a tangent along the unique geodesic from x to y.

        PT(v) = (lambda_x / lambda_y) gyr[y, -x] v. No projection,
        clipping, numerical path integration or learned connection is used.
        """
        return self._transport(x, y, vector, covector=False)

    def cotangent_transport(self, x, y, momentum):
        """Dual transport: preserves p(v), using lambda_y / lambda_x."""
        return self._transport(x, y, momentum, covector=True)
