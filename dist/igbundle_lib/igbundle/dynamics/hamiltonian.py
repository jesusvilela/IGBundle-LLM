import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
# from igbundle.geometry.riemannian import RiemannianManifold # Old generic
from igbundle.geometry.poincare import PoincareBall
# We define a dummy interface for 'RiemannianManifold' type hint if needed, or update type hints.
# For now, let's assume we use Duck Typing or just update the Type Hint.
# Let's import PoincareBall as the manifold type.

class VectorField(nn.Module):
    """
    Abstract base class for Hamiltonian Vector Fields.
    H(q, p) = T(p) + V(q)
    """
    def __init__(self, manifold: PoincareBall, potential_module: Optional[nn.Module] = None):
        super().__init__()
        self.manifold = manifold
        self.potential_module = potential_module

    def kinetic_energy(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """T(p) = 0.5 * <p, p>_q = 0.5 * p^T G^-1(q) p"""
        # G^-1(q) is the inverse metric (conformal factor for Poincare)
        # For Poincare Ball: G_ij = lambda^2 * delta_ij
        # lambda = 2 / (1 - ||q||^2)
        # G^-1 = (1/lambda)^2 * I which is scalar * Identity locally
        
        lambda_q = self.manifold.conformal_factor(q)
        inv_metric_scalar = 1.0 / (lambda_q ** 2)
        
        # p has shape [batch, ..., dim]
        # p_sq = ||p||^2_Euclidean
        p_sq = torch.sum(p * p, dim=-1)
        
        # Output shape should match p_sq's batch dims
        # inv_metric_scalar shape: [batch, ..., 1] -> squeeze to match p_sq
        return 0.5 * inv_metric_scalar.squeeze(-1) * p_sq

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """V(q) = V_learned(q)"""
        if self.potential_module is not None:
            return self.potential_module(q).squeeze(-1) # Ensure scalar shape matching batch
        return torch.zeros(q.shape[:-1], device=q.device)

    def total_energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.kinetic_energy(p, q) + self.potential_energy(q)

class LeapfrogIntegrator(nn.Module):
    """
    Symplectic Integrator for Hamiltonian Systems on Manifolds.
    Based on 'Symplectic Integration on Riemannian Manifolds'.
    """
    def __init__(self, vector_field: VectorField, 
                 step_size: float = 0.1, 
                 num_steps: int = 4):
        super().__init__()
        self.vf = vector_field
        self.step_size = step_size
        self.num_steps = num_steps
        
    def grad_V(self, q: torch.Tensor) -> torch.Tensor:
        """Compute gradient of Potential V wrt q."""
        q = q.detach().requires_grad_(True)
        v = self.vf.potential_energy(q)
        grad = torch.autograd.grad(v.sum(), q, create_graph=True)[0]
        return grad

    def step(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Leapfrog Step."""
        # Checkpointing disables grad by default in forward pass.
        # We need to explicitly enable it to compute forces via autograd.
        with torch.enable_grad():
            dt = self.step_size
            
            # 1. Momentum Half-Step: p_{t+0.5} = p_t - 0.5 * dt * dV/dq
            # Note: Theoretically should also include dT/dq if metric depends on q.
            # For Poincare, metric depends on q. 
            # T(q,p) = 0.5 * p^T G^-1(q) p
            # dT/dq term exists. But often neglected in simple "Euclidean-like" leapfrog approximations 
            # or handled by splitting.
            # Strict Manifold Leapfrog requires implicit solve or splitting.
            # We'll use the "Euclidean" approximation for momentum update but "Manifold" for position update
            # as a starting point (common in HMC on manifolds).
            # Better: p_{t+0.5} = p_t - 0.5 * dt * grad_H_q(q, p)
            
            # Calculate full grad_H wrt q
            q_in = q.detach().requires_grad_(True)
            H = self.vf.total_energy(q_in, p.detach())
            grad_H_q = torch.autograd.grad(H.sum(), q_in, create_graph=True)[0]
            
            p_half = p - 0.5 * dt * grad_H_q
            
            # 2. Position Full-Step: q_{t+1} = exp_q(dt * grad_H_p)
            # grad_H_p for T = 0.5 p^T G^-1 p is G^-1 p = v (velocity)
            # So q_{t+1} = exp(q, dt * v)
            p_half_in = p_half.detach().requires_grad_(True)
            T_half = self.vf.kinetic_energy(p_half_in, q.detach())
            grad_H_p = torch.autograd.grad(T_half.sum(), p_half_in)[0] # This is v
            
            q_new = self.vf.manifold.exp_map(q, dt * grad_H_p)
            
            # 3. Momentum Half-Step: p_{t+1} = p_{t+0.5} - 0.5 * dt * grad_H_q(q_new)
            # We need to transport p_half to q_new tangent space before updating?
            # Yes, strictly speaking. Parallel Transport required.
            # p_{t+0.5} @ q  --> PT --> p_{t+0.5} @ q_new
            p_half_transported = self.vf.manifold.parallel_transport(q, q_new, p_half)
            
            q_new_in = q_new.detach().requires_grad_(True)
            H_new = self.vf.total_energy(q_new_in, p_half_transported.detach())
            grad_H_q_new = torch.autograd.grad(H_new.sum(), q_new_in, create_graph=True)[0]
            
            p_new = p_half_transported - 0.5 * dt * grad_H_q_new
            
            return q_new, p_new

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_curr, p_curr = q, p
        for _ in range(self.num_steps):
            q_curr, p_curr = self.step(q_curr, p_curr)
        return q_curr, p_curr

# Alias for compatibility with geometric_adapter if needed
class HamiltonianSystem(VectorField):
    """Wrapper or Alias for VectorField to satisfy legacy imports."""
    pass
