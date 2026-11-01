"""Canonical Hamiltonian dynamics for the prescribed Poincare metric."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from igbundle.geometry.poincare import PoincareBall


class VectorField(nn.Module):
    """Canonical Hamiltonian H(q,p) = 1/2 p^T g(q)^-1 p + V(q)."""

    def __init__(self, manifold: PoincareBall,
                 potential_module: Optional[nn.Module] = None):
        super().__init__()
        self.manifold = manifold
        self.potential_module = potential_module

    def kinetic_energy(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """K(q,p) = ||p||^2 / (2 lambda(q)^2), with p a covector."""
        inverse_metric = self.manifold.conformal_factor(q).reciprocal().square()
        return 0.5 * inverse_metric.squeeze(-1) * p.square().sum(dim=-1)

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        if self.potential_module is None:
            return q.sum(dim=-1) * 0
        return self.potential_module(q).squeeze(-1)

    def total_energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.kinetic_energy(p, q) + self.potential_energy(q)


class LeapfrogIntegrator(nn.Module):
    """Symmetric potential/geodesic/potential split in canonical coordinates.

    This opt-in integrator uses exact geodesic flow for the kinetic substep.
    Momentum is a covector and is dual-parallel-transported. Finite-step and
    finite-precision energy error remain; this does not certify model benefit.
    """

    def __init__(self, vector_field: VectorField,
                 step_size: float = 0.1,
                 num_steps: int = 4):
        super().__init__()
        if (isinstance(step_size, bool) or not isinstance(step_size, (int, float))
                or not math.isfinite(step_size)):
            raise ValueError("step_size must be finite")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        self.vf = vector_field
        self.step_size = float(step_size)
        self.num_steps = num_steps

    def _validate_state(self, q: torch.Tensor, p: torch.Tensor) -> None:
        dim = self.vf.manifold.dim
        if (not isinstance(q, torch.Tensor) or not isinstance(p, torch.Tensor)
                or q.ndim < 1 or q.shape[-1] != dim or q.shape != p.shape
                or not q.is_floating_point() or not p.is_floating_point()
                or q.dtype != p.dtype or q.device != p.device):
            raise ValueError("q and p must be same-shaped floating tensors ending in manifold.dim")
        if q.dtype not in (torch.float32, torch.float64):
            raise ValueError("Hamiltonian integration requires float32 or float64 states")
        if not torch.isfinite(q).all() or not torch.isfinite(p).all():
            raise ValueError("q and p must be finite")
        if (self.vf.manifold.c * q.square().sum(dim=-1) >= 1).any():
            raise ValueError("q must lie strictly inside the curvature-scaled Poincare ball")

    def _potential_gradient(self, q: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        q_eval = q if q.requires_grad else q.detach().requires_grad_(True)
        potential = self.vf.potential_energy(q_eval)
        if not potential.requires_grad:
            return torch.zeros_like(q_eval)
        return torch.autograd.grad(
            potential.sum(), q_eval, create_graph=create_graph,
        )[0]

    def grad_V(self, q: torch.Tensor) -> torch.Tensor:
        """Compute the coordinate covector dV/dq."""
        create_graph = torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            if torch.is_inference(q):
                q = q.clone()
            return self._potential_gradient(q, create_graph=create_graph)

    def _step(self, q: torch.Tensor, p: torch.Tensor,
              *, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = self.step_size
        with torch.inference_mode(False), torch.enable_grad():
            if torch.is_inference(q):
                q = q.clone()
            if torch.is_inference(p):
                p = p.clone()

            p_half = p - 0.5 * dt * self._potential_gradient(
                q, create_graph=create_graph,
            )
            velocity = p_half / self.vf.manifold.conformal_factor(q).square()
            q_new = self.vf.manifold.exp_map(q, dt * velocity)
            p_new = self.vf.manifold.cotangent_transport(q, q_new, p_half)
            p_new = p_new - 0.5 * dt * self._potential_gradient(
                q_new, create_graph=create_graph,
            )

        if (not torch.isfinite(q_new).all() or not torch.isfinite(p_new).all()
                or (self.vf.manifold.c * q_new.square().sum(dim=-1) >= 1).any()):
            raise RuntimeError("Hamiltonian step left the finite Poincare phase space")
        return q_new, p_new

    def step(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance one symmetric potential/geodesic/potential split step."""
        self._validate_state(q, p)
        return self._step(q, p, create_graph=torch.is_grad_enabled())

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_state(q, p)
        create_graph = torch.is_grad_enabled()
        q_curr, p_curr = q, p
        for _ in range(self.num_steps):
            q_curr, p_curr = self._step(q_curr, p_curr, create_graph=create_graph)
        return q_curr, p_curr


class HamiltonianSystem(VectorField):
    """Compatibility name for the canonical Poincare Hamiltonian."""
