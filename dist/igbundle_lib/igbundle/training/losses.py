import torch
import torch.nn as nn
from igbundle.dynamics.hamiltonian import LeapfrogIntegrator, VectorField
# from igbundle.geometry.riemannian import RiemannianManifold # Unused

class RetrospectionLoss(nn.Module):
    """
    Enforces Time-Reversal Symmetry on the reasoning process.
    L_retro = ||q_0 - q_0'||^2 + ||p_0 - p_0'||^2
    where (q_0', p_0') is the result of forward-then-backward simulation.
    """
    def __init__(self, integrator: LeapfrogIntegrator, lambda_reg: float = 0.5):
        super().__init__()
        self.integrator = integrator
        self.lambda_reg = lambda_reg
        
    def forward(self, q_init: torch.Tensor, p_init: torch.Tensor) -> torch.Tensor:
        # 1. Forward Simulation (The "Thought")
        # We assume gradients flow through this for other losses, 
        # but for Retrospection, we might want to check if the STRUCTURE allows reversibility.
        q_T, p_T = self.integrator(q_init, p_init)
        
        # 2. Reverse Momentum (Time Reversal)
        p_T_rev = -p_T
        
        # 3. Backward Simulation (The "Retrospection")
        # We want to reconstruct the initial state
        q_0_recon, p_0_recon_rev = self.integrator(q_T, p_T_rev)
        
        # 4. Compare
        # p_0_recon_rev corresponds to -p_0 in the ideal case
        p_0_recon = -p_0_recon_rev
        
        # Calculate distance on manifold for q
        # For simplicity in v1, use Euclidean or Hyperbolic dist if available
        # Manifold.distance(q1, q2)
        dist_sq_q = self.integrator.vf.manifold.distance(q_init, q_0_recon) ** 2
        
        # Euclidean distance for Tangent Space momentum
        dist_sq_p = torch.sum((p_init - p_0_recon) ** 2, dim=-1)
        
        loss = dist_sq_q + dist_sq_p
        
        return self.lambda_reg * loss.mean()
