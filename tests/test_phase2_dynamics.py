import torch
import unittest
from igbundle.geometry.poincare import PoincareBall
from igbundle.dynamics.hamiltonian import LeapfrogIntegrator, VectorField
import torch.nn as nn
# from igbundle.training.losses import RetrospectionLoss # Avoid import hell

class RetrospectionLoss(nn.Module):
    """
    Enforces Time-Reversal Symmetry on the reasoning process.
    L_retro = ||q_0 - q_0'||^2 + ||p_0 - p_0'||^2
    """
    def __init__(self, integrator, lambda_reg: float = 0.5):
        super().__init__()
        self.integrator = integrator
        self.lambda_reg = lambda_reg
        
    def forward(self, q_init: torch.Tensor, p_init: torch.Tensor) -> torch.Tensor:
        q_T, p_T = self.integrator(q_init, p_init)
        p_T_rev = -p_T
        q_0_recon, p_0_recon_rev = self.integrator(q_T, p_T_rev)
        p_0_recon = -p_0_recon_rev
        
        # dist_sq_q = self.integrator.vf.manifold.distance(q_init, q_0_recon) ** 2
        # Use simple Euclidean for stability in this test if manifold dist not robust or expensive
        dist_sq_q = torch.sum((q_init - q_0_recon)**2, dim=-1)
        dist_sq_p = torch.sum((p_init - p_0_recon) ** 2, dim=-1)
        loss = dist_sq_q + dist_sq_p
        return self.lambda_reg * loss.mean()

class SimpleHarmonicOscillator(VectorField):
    """
    V(q) = 0.5 * k * ||q||^2
    Simple potential to test dynamics.
    """
    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        # Harmonic potential in Euclidean sense, but q is on Poincare Ball.
        # Let's use simple squared norm (within ball < 1).
        q_norm_sq = torch.sum(q*q, dim=-1)
        return 0.5 * 10.0 * q_norm_sq # k=10

class TestPhase2(unittest.TestCase):
    def test_dynamics(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dim = 16
        
        # 1. Setup
        manifold = PoincareBall(dim=dim, c=1.0).to(device)
        vf = SimpleHarmonicOscillator(manifold).to(device)
        integrator = LeapfrogIntegrator(vf, step_size=0.01, num_steps=10).to(device)
        
        # 2. Init State (Small q, random p)
        q0 = torch.randn(2, dim, device=device) * 0.1
        q0.requires_grad_(True)
        p0 = torch.randn(2, dim, device=device) * 0.1
        
        # 3. Energy Check (Conservation)
        E_start = vf.total_energy(q0, p0).mean()
        
        # 4. Integrate Forward
        qT, pT = integrator(q0, p0)
        E_end = vf.total_energy(qT, pT).mean()
        
        drift = abs(E_end.item() - E_start.item())
        print(f"Energy Drift: {drift:.6f}")
        self.assertLess(drift, 0.05, "Hamiltonian should be roughly conserved")
        
        # 5. Retrospection (Reversibility)
        retro_loss_fn = RetrospectionLoss(integrator)
        loss = retro_loss_fn(q0, p0)
        print(f"Retrospection Loss: {loss.item():.6f}")
        
        # Loss should be low for symplectic integrator on short trajectory
        self.assertLess(loss.item(), 0.01, "Dynamics should be reversible")
        
        # 6. Backward Pass (Gradient Check)
        loss.backward()
        self.assertIsNotNone(q0.grad, "Gradients must flow to q0")

if __name__ == '__main__':
    unittest.main()
