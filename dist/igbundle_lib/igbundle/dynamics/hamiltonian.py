import torch
import torch.nn as nn
from typing import Tuple

class HamiltonianSystem(nn.Module):
    """
    Implements Hamiltonian Dynamics on the latent manifold.
    
    H(q, p) = T(p) + V(q)
    
    Where:
    - q: Position (Latent State on Manifold)
    - p: Momentum (Rate of Reasoning Change)
    - T(p): Kinetic Energy (Cost of changing thought)
    - V(q): Potential Energy (Semantic Stability/Energy Landscape)
    """
    def __init__(self, hidden_dim: int, mass: float = 1.0):
        super().__init__()
        self.mass = mass
        
        # Learnable Potential Energy Function V(q)
        # This represents the "Energy Landscape" of meaning.
        # Low energy = Stable/True concepts. High energy = Contradiction/Nonsense.
        self.potential_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1) # Scalar Energy
        )

    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """T(p) = 0.5 * p^T * M^-1 * p"""
        # Simplification: scalar mass M=1
        return 0.5 * torch.sum(p**2, dim=-1) / self.mass

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """V(q) learned from data + Barrier for Manifold."""
        # 1. Learned Topology
        v_net = self.potential_net(q).squeeze(-1)
        
        # 2. Horizon Barrier (Poincaré Ball boundary)
        # Prevent escaping the universe (|q| -> 1)
        # Use log-barrier: -log(1 - |q|^2) approaches inf as |q|->1
        # Clamp to avoid nan at q=0 or q=1 during calc (though q in open ball)
        q_norm_sq = torch.sum(q**2, dim=-1).clamp(min=1e-6, max=0.999)
        v_barrier = -0.1 * torch.log(1.0 - q_norm_sq)
        
        return v_net + v_barrier

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """H(q, p) = T(p) + V(q)"""
        return self.kinetic_energy(p) + self.potential_energy(q)

    def symplectic_step(self, q: torch.Tensor, p: torch.Tensor, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog integration step (Symplectic - preserves phase space volume).
        
        Updates (q, p) -> (q_new, p_new) over time dt.
        """
        # 1. Update Momentum (half step) using Force (-dV/dq)
        # We need grad of V(q) w.r.t q
        # Since we are inside a computation graph, we can use autograd
        
        # Enable grad for q temporarily if not already
        with torch.enable_grad():
             q_in = q.detach().requires_grad_(True)
             v = self.potential_energy(q_in)
             # Force = -dV/dq
             grads = torch.autograd.grad(v.sum(), q_in, create_graph=True)[0]
             force = -grads
             
        p_half = p + 0.5 * dt * force
        
        # 2. Update Position (full step) using Momentum
        # dq/dt = p/m
        q_new = q + dt * (p_half / self.mass)
        
        # 3. Update Momentum (half step) using new Force
        with torch.enable_grad():
             q_new_in = q_new.detach().requires_grad_(True)
             v_new = self.potential_energy(q_new_in)
             grads_new = torch.autograd.grad(v_new.sum(), q_new_in, create_graph=True)[0]
             force_new = -grads_new
             
        p_new = p_half + 0.5 * dt * force_new
        
        return q_new, p_new
