"""
Test Geometric Scrambling Diagnostic
=====================================

Validates the classical OTOC analog and scrambling time measurement
against the Phase 2 Hamiltonian dynamics.

Key metrics:
- Scrambling time τ_s: When information spreads across all bundles
- Lyapunov exponent λ: Rate of exponential divergence  
- OTOC analog: Cross-fiber influence matrix
- Gibbs temperature β: Coherence Threshold

Author: Jesús Vilela Jato
Date: February 2026
"""

import torch
import math
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

# =============================================================================
# Minimal Phase 2 Components for Testing
# =============================================================================

@dataclass
class Phase2Config:
    """Configuration for Phase 2 Hamiltonian training."""
    latent_dim: int = 64
    num_fibers: int = 16
    curvature: float = -1.0
    max_norm: float = 0.95
    num_leapfrog_steps: int = 4
    step_size: float = 0.1
    damping: float = 0.01
    temperature: float = 1.0


class PoincareBallOps:
    """Riemannian operations on the Poincaré ball."""
    
    def __init__(self, curvature: float = -1.0, eps: float = 1e-7):
        self.c = abs(curvature)
        self.eps = eps
    
    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        norm_sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(max=1 - self.eps)
        return 2.0 / (1.0 - self.c * norm_sq + self.eps)
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        
        c = self.c
        num = (1 + 2*c*xy_dot + c*y_norm_sq) * x + (1 - c*x_norm_sq) * y
        denom = 1 + 2*c*xy_dot + c*c*x_norm_sq*y_norm_sq + self.eps
        
        return num / denom
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        lambda_x = self._lambda_x(x)
        
        coef = torch.tanh(sqrt_c * lambda_x * v_norm / 2) / (sqrt_c * v_norm + self.eps)
        direction = v * coef
        
        return self.mobius_add(x, direction)
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sqrt_c = math.sqrt(self.c)
        diff = self.mobius_add(-x, y)
        norm_diff = torch.norm(diff, dim=-1, keepdim=True).clamp(max=1-self.eps)
        
        return 2 * torch.atanh(sqrt_c * norm_diff) / sqrt_c
    
    def project_to_ball(self, x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * torch.clamp(max_norm / (norm + self.eps), max=1.0)


class HamiltonianSystemTest:
    """Simplified Hamiltonian system for scrambling test."""
    
    def __init__(self, config: Phase2Config, manifold: PoincareBallOps):
        self.config = config
        self.manifold = manifold
        self.mass = torch.ones(config.latent_dim)
        self.inv_mass = 1.0 / self.mass
    
    def total_potential(self, q: torch.Tensor) -> torch.Tensor:
        """Combined geometric + semantic potential."""
        norm_sq = torch.sum(q * q, dim=-1)
        r = torch.sqrt(norm_sq + 1e-7)
        
        # Geometric potential
        origin_penalty = -torch.log(r + 0.1)
        boundary_penalty = 10.0 * torch.relu(r - 0.9) ** 2
        
        # Simple semantic potential (quadratic well)
        semantic = 0.1 * norm_sq
        
        return origin_penalty + boundary_penalty + semantic
    
    def compute_forces(self, q: torch.Tensor) -> torch.Tensor:
        """Compute forces F = -∇V(q) via autograd."""
        q_grad = q.detach().clone().requires_grad_(True)
        V = self.total_potential(q_grad)
        V_sum = V.sum()
        
        grad_V = torch.autograd.grad(V_sum, q_grad)[0]
        
        return -grad_V
    
    def leapfrog_step(
        self, 
        q: torch.Tensor, 
        p: torch.Tensor, 
        dt: float,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single leapfrog integration step."""
        sign = -1.0 if reverse else 1.0
        dt_eff = sign * dt
        damping = self.config.damping
        
        F = self.compute_forces(q)
        p_half = p + (dt_eff / 2) * F - damping * dt_eff * p
        
        velocity = p_half * self.inv_mass
        q_new = self.manifold.exp_map(q, dt_eff * velocity)
        q_new = self.manifold.project_to_ball(q_new, self.config.max_norm)
        
        F_new = self.compute_forces(q_new)
        p_new = p_half + (dt_eff / 2) * F_new - damping * dt_eff * p_half
        
        return q_new, p_new
    
    def gibbs_temperature(self) -> float:
        """Effective inverse temperature β from damping."""
        q = self.config.damping
        if q > 0 and q < 1:
            return -math.log(q / (1 - q))
        return float('inf')
    
    def geometric_scrambling_time(
        self,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        perturbation_scale: float = 1e-6,
        max_steps: int = 50,
        saturation_threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Measure information spreading rate via geodesic divergence."""
        with torch.no_grad():
            perturbation = torch.randn_like(q_init) * perturbation_scale
            
            q_orig = q_init.clone()
            p_orig = p_init.clone()
            q_pert = self.manifold.project_to_ball(q_init + perturbation, self.config.max_norm)
            p_pert = p_init.clone()
            
            distances = []
            dt = self.config.step_size
            
            for step in range(max_steps):
                d_hyp = self.manifold.distance(q_orig, q_pert)
                mean_dist = d_hyp.mean().item()
                distances.append(mean_dist)
                
                q_orig, p_orig = self.leapfrog_step(q_orig, p_orig, dt, reverse=False)
                q_pert, p_pert = self.leapfrog_step(q_pert, p_pert, dt, reverse=False)
            
            distances = torch.tensor(distances)
            max_dist = distances.max().item()
            threshold = saturation_threshold * max_dist
            
            scrambling_time = max_steps
            saturated = False
            for t, d in enumerate(distances):
                if d >= threshold:
                    scrambling_time = t
                    saturated = True
                    break
            
            early_steps = min(10, len(distances) - 1)
            if distances[0] > 0 and distances[early_steps] > distances[0]:
                lyapunov = math.log(distances[early_steps].item() / distances[0].item()) / (early_steps * dt)
            else:
                lyapunov = 0.0
            
            return {
                'scrambling_time': scrambling_time,
                'lyapunov_estimate': lyapunov,
                'distance_trajectory': distances.tolist(),
                'saturated': saturated,
                'max_distance': max_dist,
                'gibbs_beta': self.gibbs_temperature()
            }
    
    def compute_otoc_analog(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        operator_indices: Optional[List[int]] = None,
        num_steps: int = 20
    ) -> torch.Tensor:
        """Classical OTOC analog measuring cross-fiber influence."""
        batch, num_fibers, dim = q.shape
        device = q.device
        
        if operator_indices is None:
            operator_indices = list(range(num_fibers))
        
        num_ops = len(operator_indices)
        otoc_values = torch.zeros(num_steps, num_ops, num_ops, device=device)
        
        with torch.no_grad():
            dt = self.config.step_size
            eps = 1e-5
            
            for step in range(num_steps):
                for i_idx, i in enumerate(operator_indices):
                    for j_idx, j in enumerate(operator_indices):
                        q_pert = q.clone()
                        q_pert[:, j, :] += eps * torch.randn(batch, dim, device=device)
                        q_pert = self.manifold.project_to_ball(q_pert, self.config.max_norm)
                        
                        q_evolved, _ = self.leapfrog_step(q, p, dt)
                        q_pert_evolved, _ = self.leapfrog_step(q_pert, p, dt)
                        
                        diff_i = (q_evolved[:, i, :] - q_pert_evolved[:, i, :]).norm(dim=-1)
                        otoc_values[step, i_idx, j_idx] = (diff_i / eps).mean()
                
                q, p = self.leapfrog_step(q, p, dt)
        
        return otoc_values


def test_scrambling_diagnostic():
    """Test the geometric scrambling diagnostic."""
    
    print("=" * 70)
    print("GEOMETRIC SCRAMBLING DIAGNOSTIC TEST")
    print("=" * 70)
    
    config = Phase2Config(
        latent_dim=64,
        num_fibers=16,
        curvature=-1.0,
        damping=0.01,
        num_leapfrog_steps=4,
        step_size=0.1
    )
    
    manifold = PoincareBallOps(curvature=config.curvature)
    hamiltonian = HamiltonianSystemTest(config, manifold)
    
    batch_size = 4
    q_init = 0.3 * torch.randn(batch_size, config.num_fibers, config.latent_dim)
    q_init = manifold.project_to_ball(q_init, config.max_norm)
    p_init = torch.randn_like(q_init) * 0.1
    
    print(f"\n[1] GIBBS TEMPERATURE")
    print("-" * 40)
    beta = hamiltonian.gibbs_temperature()
    print(f"Damping parameter q: {config.damping}")
    print(f"Effective inverse temperature β: {beta:.4f}")
    print(f"Coherence threshold: β > 1.87")
    print(f"Status: {'ABOVE THRESHOLD ✓' if beta > 1.87 else 'BELOW THRESHOLD ✗'}")
    
    print(f"\n[2] GEOMETRIC SCRAMBLING TIME")
    print("-" * 40)
    scrambling_result = hamiltonian.geometric_scrambling_time(
        q_init, p_init,
        perturbation_scale=1e-6,
        max_steps=50,
        saturation_threshold=0.95
    )
    
    print(f"Scrambling time τ_s: {scrambling_result['scrambling_time']} steps")
    print(f"Lyapunov exponent λ: {scrambling_result['lyapunov_estimate']:.6f}")
    print(f"Max distance reached: {scrambling_result['max_distance']:.6f}")
    print(f"Saturated: {scrambling_result['saturated']}")
    print(f"Gibbs β (embedded): {scrambling_result['gibbs_beta']:.4f}")
    
    distances = scrambling_result['distance_trajectory']
    print(f"\nDistance trajectory (first 20 steps):")
    max_d = max(distances[:20]) if len(distances) > 20 else max(distances)
    for i, d in enumerate(distances[:20]):
        bar_len = int(40 * d / (max_d + 1e-7))
        print(f"  t={i:2d}: {'█' * bar_len} {d:.6f}")
    
    print(f"\n[3] OTOC ANALOG (Cross-Fiber Influence)")
    print("-" * 40)
    
    test_indices = [0, 4, 8, 12]
    otoc = hamiltonian.compute_otoc_analog(
        q_init, p_init,
        operator_indices=test_indices,
        num_steps=10
    )
    
    print(f"OTOC shape: {otoc.shape}")
    print(f"Final time step cross-fiber influence matrix:")
    print(f"       Fiber 0   Fiber 4   Fiber 8   Fiber 12")
    for i, idx in enumerate(test_indices):
        row = otoc[-1, i, :].tolist()
        print(f"  F{idx:2d}:  " + "  ".join([f"{v:7.4f}" for v in row]))
    
    final_otoc = otoc[-1]
    off_diag_mask = ~torch.eye(len(test_indices), dtype=bool)
    off_diag_mean = final_otoc[off_diag_mask].mean().item()
    diag_mean = final_otoc.diag().mean().item()
    
    print(f"\nDiagonal mean (self-influence): {diag_mean:.6f}")
    print(f"Off-diagonal mean (cross-influence): {off_diag_mean:.6f}")
    print(f"Scrambling ratio (off/diag): {off_diag_mean / (diag_mean + 1e-7):.4f}")
    
    print(f"\n" + "=" * 70)
    print("SCRAMBLING DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'latent_dim': config.latent_dim,
            'num_fibers': config.num_fibers,
            'damping': config.damping,
            'curvature': config.curvature
        },
        'results': {
            'gibbs_beta': beta,
            'above_rw_threshold': beta > 1.87,
            'scrambling_time': scrambling_result['scrambling_time'],
            'lyapunov_exponent': scrambling_result['lyapunov_estimate'],
            'otoc_scrambling_ratio': off_diag_mean / (diag_mean + 1e-7),
        },
        'interpretation': {
            'quantum_advantage_regime': beta > 1.87,
            'chaotic_dynamics': scrambling_result['lyapunov_estimate'] > 0,
            'information_spreading': off_diag_mean > 0.01,
        }
    }
    
    print(json.dumps(summary, indent=2))
    
    # Save results
    output_path = r'H:\LLM-MANIFOLD\igbundle-llm\scrambling_diagnostic_results.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return summary


if __name__ == "__main__":
    test_scrambling_diagnostic()
