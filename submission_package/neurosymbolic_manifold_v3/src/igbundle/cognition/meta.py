import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

from ..dynamics.hamiltonian import HamiltonianSystem
from ..geometry.poincare import PoincareBall

class MetaCognitiveLoop(nn.Module):
    """
    System 2 Meta-Cognitive Loop.
    
    Verifies and refines thoughts based on the learned Semantic Potential V(q).
    
    Logic:
    1. Verify: Calculate Energy E = V(q).
    2. Decision: If E > threshold, trigger 'Refinement'.
    3. Refine: Perform Gradient Descent on V(q) to minimize Implausibility.
    """
    def __init__(self, vector_field: HamiltonianSystem, threshold: float = 0.5, refine_steps: int = 5, lr: float = 0.1):
        super().__init__()
        self.vf = vector_field # Provides potential_energy(q) and manifold ops
        self.threshold = threshold
        self.refine_steps = refine_steps
        self.lr = lr
        
    def verify(self, q: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Check if thought is valid.
        Returns (Energy, IsValid)
        """
        with torch.enable_grad():
            if not q.requires_grad:
                q = q.detach().requires_grad_(True)
            v = self.vf.potential_energy(q)
            
        is_valid = (v < self.threshold)
        return v, is_valid

    def refine(self, q: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Refine thought q by minimizing V(q).
        Returns (q_refined, energy_trace)
        """
        # Ensure q is on manifold and ready for opt
        q_curr = q.clone().detach().requires_grad_(True)
        energy_trace = []
        
        # Optimization Loop (Gradient Descent on Manifold)
        # q_{t+1} = exp_q(-lr * grad_V)
        
        for k in range(self.refine_steps):
            with torch.enable_grad():
                v = self.vf.potential_energy(q_curr)
                energy_trace.append(v.mean().item()) # efficient logging
                
                # Compute Gradient
                grad_v = torch.autograd.grad(v.sum(), q_curr)[0]
                
            # Manifold Update
            # direction = -grad_v (Riemannian Gradient? For now use Euclidean approx projected)
            # Strictly: grad_R = G^-1 grad_E
            # Poincare G^-1 = (1/lambda)^2 * I.
            # So just scaling.
            
            # Simple version: Retraction
            # q_new = q - lr * grad_v <-- This is dangerous on Poincare edge
            # Use ExpMap: q_new = exp(q, -lr * grad_v_riemannian)
            
            # Get Riemannian Gradient
            scaling = self.vf.manifold.conformal_factor(q_curr).unsqueeze(-1) ** (-2)
            grad_riemann = grad_v * scaling
            
            # Update
            # We use projected gradient descent if exact exp map is expensive, but we have exp_map
            step = -self.lr * grad_riemann
            q_next = self.vf.manifold.exp_map(q_curr, step)
            
            q_curr = q_next.detach().requires_grad_(True)
            
        final_v = self.vf.potential_energy(q_curr).mean().item()
        energy_trace.append(final_v)
        
        return q_curr, energy_trace

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Process thought q through the loop.
        """
        energy, valid_mask = self.verify(q)
        # valid_mask is per item in batch.
        # We refine only invalid ones? Or all?
        # For simplicity, if ANY is invalid in batch (or just refining all for better quality), let's refine all.
        # Ideally handle batch masking.
        
        # For demo/prototype: If huge energy, refine.
        avg_energy = energy.mean()
        info = {"initial_energy": avg_energy.item(), "refined": False, "trace": []}
        
        if avg_energy > self.threshold:
            q_refined, trace = self.refine(q)
            info["refined"] = True
            info["trace"] = trace
            info["final_energy"] = trace[-1]
            return q_refined, info
            
        return q, info
