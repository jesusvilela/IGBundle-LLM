"""
Geometric Scrambling Diagnostic for ManifoldGL Phase 2.5

Implements classical analogs of quantum scrambling measures:
- Out-of-Time-Order Correlators (OTOC) via geodesic divergence
- Scrambling time τ_s via distance saturation
- Lyapunov exponent estimation from hyperbolic flow

Theory: In hyperbolic geometry (Poincaré ball), geodesics diverge exponentially.
This mirrors quantum scrambling where localized information spreads across
all degrees of freedom. We measure this WITHOUT quantum hardware by tracking
how perturbations propagate through Hamiltonian flow on the manifold.

Reference: Rajakumar & Watson (2026) - Gibbs Sampling Quantum Advantage
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import math

@dataclass
class ScramblingResult:
    """Container for scrambling analysis results."""
    scrambling_time: float          # τ_s: time to saturation
    lyapunov_exponent: float        # λ: rate of divergence
    saturation_distance: float      # d_sat: final hyperbolic distance
    distance_trajectory: List[float] # d(t) over time
    is_chaotic: bool                # True if λ > 0
    effective_temperature: float    # β from damping parameter
    above_hardness_threshold: bool  # β > 1.87 (Rajakumar-Watson)


class GeometricScramblingDiagnostic(nn.Module):
    """
    Measures information spreading in Hamiltonian dynamics on Poincaré ball.
    
    Classical analog of quantum scrambling:
    - OTOC ↔ Geodesic divergence rate
    - Scrambling time ↔ Distance saturation time
    - Quantum advantage threshold ↔ β > 1.87
    """
    
    def __init__(
        self,
        manifold,  # PoincareBall instance
        latent_dim: int = 64,
        perturbation_scale: float = 1e-4,
        max_steps: int = 100,
        step_size: float = 0.05,
        damping: float = 0.01,
        saturation_threshold: float = 0.95
    ):
        super().__init__()
        self.manifold = manifold
        self.latent_dim = latent_dim
        self.perturbation_scale = perturbation_scale
        self.max_steps = max_steps
        self.step_size = step_size
        self.damping = damping
        self.saturation_threshold = saturation_threshold
        
        # Learnable potential for energy landscape
        self.potential_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1)
        )
        
    def gibbs_temperature(self) -> float:
        """
        Effective inverse temperature from damping parameter.
        Per Rajakumar & Watson (2026), β > 1.87 implies classical hardness.
        
        Mapping: q = e^{-β}/(1 + e^{-β}) → β = -log(q/(1-q))
        """
        q = self.damping
        if q > 0 and q < 1:
            return -math.log(q / (1 - q))
        return float('inf')
    
    def is_above_hardness_threshold(self) -> bool:
        """Check if operating above quantum advantage regime."""
        return self.gibbs_temperature() > 1.87
    
    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance in Poincaré ball.
        d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
        """
        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
        x_norm_sq = torch.sum(x ** 2, dim=-1)
        y_norm_sq = torch.sum(y ** 2, dim=-1)
        
        # Clamp to avoid numerical issues at boundary
        x_norm_sq = torch.clamp(x_norm_sq, max=0.999)
        y_norm_sq = torch.clamp(y_norm_sq, max=0.999)
        
        numerator = 2 * diff_norm_sq
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
        
        argument = 1 + numerator / (denominator + 1e-8)
        return torch.acosh(torch.clamp(argument, min=1.0 + 1e-8))
    
    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """Semantic potential energy V(q)."""
        return self.potential_net(q).squeeze(-1)
    
    def kinetic_energy(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Kinetic energy T(p) = 0.5 * p^T G^{-1}(q) p
        For Poincaré ball: G = λ² I where λ = 2/(1-||q||²)
        """
        q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)
        q_norm_sq = torch.clamp(q_norm_sq, max=0.999)
        lambda_q = 2.0 / (1 - q_norm_sq)
        inv_metric = 1.0 / (lambda_q ** 2)
        
        p_norm_sq = torch.sum(p ** 2, dim=-1, keepdim=True)
        return 0.5 * (inv_metric * p_norm_sq).squeeze(-1)
    
    def total_energy(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Hamiltonian H(q,p) = T(p) + V(q)."""
        return self.kinetic_energy(p, q) + self.potential_energy(q)
    
    def leapfrog_step(
        self, 
        q: torch.Tensor, 
        p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single symplectic leapfrog step with damping.
        Damping breaks time-reversal symmetry but stabilizes flow.
        """
        dt = self.step_size
        
        with torch.enable_grad():
            # Gradient of potential
            q_grad = q.detach().requires_grad_(True)
            V = self.potential_energy(q_grad)
            grad_V = torch.autograd.grad(V.sum(), q_grad)[0]
        
        # Momentum half-step
        p_half = p - 0.5 * dt * grad_V
        
        # Apply damping (breaks symplecticity but adds stability)
        p_half = p_half * (1 - self.damping)
        
        # Position full-step via exponential map
        # Velocity = G^{-1} p (inverse metric applied to momentum)
        q_norm_sq = torch.sum(q ** 2, dim=-1, keepdim=True)
        q_norm_sq = torch.clamp(q_norm_sq, max=0.999)
        lambda_q = 2.0 / (1 - q_norm_sq)
        velocity = p_half / (lambda_q ** 2)
        
        # Exponential map on Poincaré ball
        q_new = self._exp_map(q, dt * velocity)
        
        # Momentum half-step at new position
        with torch.enable_grad():
            q_new_grad = q_new.detach().requires_grad_(True)
            V_new = self.potential_energy(q_new_grad)
            grad_V_new = torch.autograd.grad(V_new.sum(), q_new_grad)[0]
        
        p_new = p_half - 0.5 * dt * grad_V_new
        
        return q_new, p_new
    
    def _exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on Poincaré ball.
        exp_x(v) = x ⊕ tanh(λ_x ||v|| / 2) * v / ||v||
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-8
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        x_norm_sq = torch.clamp(x_norm_sq, max=0.999)
        lambda_x = 2.0 / (1 - x_norm_sq)
        
        # Direction
        v_normalized = v / v_norm
        
        # Magnitude after exp map
        tanh_arg = lambda_x * v_norm / 2
        tanh_arg = torch.clamp(tanh_arg, max=15.0)  # Numerical stability
        magnitude = torch.tanh(tanh_arg)
        
        # Möbius addition: x ⊕ y
        y = magnitude * v_normalized
        return self._mobius_add(x, y)
    
    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in Poincaré ball.
        x ⊕ y = ((1 + 2<x,y> + ||y||²)x + (1 - ||x||²)y) / 
                (1 + 2<x,y> + ||x||²||y||²)
        """
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2 * xy_dot + y_norm_sq) * x + (1 - x_norm_sq) * y
        denominator = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq
        
        result = numerator / (denominator + 1e-8)
        
        # Project back to ball if numerical drift
        result_norm = torch.norm(result, dim=-1, keepdim=True)
        result = torch.where(
            result_norm > 0.99,
            result * 0.99 / (result_norm + 1e-8),
            result
        )
        return result
    
    def measure_scrambling(
        self,
        initial_state: Optional[torch.Tensor] = None,
        perturbation_direction: Optional[torch.Tensor] = None
    ) -> ScramblingResult:
        """
        Measure geometric scrambling by tracking perturbation divergence.
        
        Returns ScramblingResult with:
        - scrambling_time: when distance saturates
        - lyapunov_exponent: rate of exponential divergence
        - distance_trajectory: full d(t) curve
        """
        device = next(self.parameters()).device
        
        # Initialize state at origin if not provided
        if initial_state is None:
            initial_state = torch.zeros(1, self.latent_dim, device=device)
        
        # Random perturbation direction if not provided
        if perturbation_direction is None:
            perturbation_direction = torch.randn(1, self.latent_dim, device=device)
            perturbation_direction = perturbation_direction / (
                torch.norm(perturbation_direction) + 1e-8
            )
        
        # Initial states
        q0 = initial_state.clone()
        q_pert = initial_state + self.perturbation_scale * perturbation_direction
        
        # Project perturbed state into ball
        q_pert_norm = torch.norm(q_pert, dim=-1, keepdim=True)
        q_pert = torch.where(
            q_pert_norm > 0.99,
            q_pert * 0.99 / (q_pert_norm + 1e-8),
            q_pert
        )
        
        # Initial momenta (zero = start from rest)
        p0 = torch.zeros_like(q0)
        p_pert = torch.zeros_like(q_pert)
        
        # Track distances
        distances = []
        energies = []
        
        # Evolve both trajectories
        for step in range(self.max_steps):
            # Compute hyperbolic distance
            d = self.hyperbolic_distance(q0, q_pert)
            distances.append(d.item())
            
            # Track energy conservation
            E = self.total_energy(q0, p0)
            energies.append(E.item())
            
            # Evolve
            q0, p0 = self.leapfrog_step(q0, p0)
            q_pert, p_pert = self.leapfrog_step(q_pert, p_pert)
        
        # Analyze results
        distances = np.array(distances)
        
        # Find scrambling time (when distance reaches saturation threshold of max)
        max_dist = np.max(distances)
        saturation_level = self.saturation_threshold * max_dist
        
        scrambling_time = self.max_steps  # Default if never saturates
        for t, d in enumerate(distances):
            if d >= saturation_level:
                scrambling_time = t
                break
        
        # Estimate Lyapunov exponent from early exponential growth
        # d(t) ≈ d(0) * exp(λt) → λ = (1/t) * log(d(t)/d(0))
        early_window = min(20, len(distances) // 2)
        if distances[0] > 1e-10 and early_window > 1:
            log_ratio = np.log(distances[early_window] / (distances[0] + 1e-10))
            lyapunov = log_ratio / (early_window * self.step_size)
        else:
            lyapunov = 0.0
        
        # Effective temperature
        beta = self.gibbs_temperature()
        
        return ScramblingResult(
            scrambling_time=scrambling_time * self.step_size,
            lyapunov_exponent=lyapunov,
            saturation_distance=max_dist,
            distance_trajectory=distances.tolist(),
            is_chaotic=(lyapunov > 0.01),
            effective_temperature=beta,
            above_hardness_threshold=(beta > 1.87)
        )
    
    def otoc_analog(
        self,
        operator_W: torch.Tensor,
        operator_V: torch.Tensor,
        initial_state: torch.Tensor,
        time_steps: int = 50
    ) -> List[float]:
        """
        Classical analog of Out-of-Time-Order Correlator.
        
        OTOC: C(t) = <[W(t), V(0)]²>
        
        Classical analog: We measure how perturbation in W-direction
        at time t affects V-direction measurement.
        
        Returns C(t) trajectory.
        """
        device = initial_state.device
        
        # Normalize operators to unit vectors
        W = operator_W / (torch.norm(operator_W) + 1e-8)
        V = operator_V / (torch.norm(operator_V) + 1e-8)
        
        otoc_values = []
        
        q = initial_state.clone()
        p = torch.zeros_like(q)
        
        for t in range(time_steps):
            # Evolve forward to time t
            q_t, p_t = q.clone(), p.clone()
            for _ in range(t):
                q_t, p_t = self.leapfrog_step(q_t, p_t)
            
            # Perturb in W direction at time t
            q_perturbed = q_t + self.perturbation_scale * W
            
            # Evolve backward (negate momentum for time reversal)
            p_back = -p_t.clone()
            q_back = q_perturbed.clone()
            for _ in range(t):
                q_back, p_back = self.leapfrog_step(q_back, p_back)
            
            # Measure deviation in V direction
            # This is the "commutator" analog: how much does W(t) not commute with V(0)?
            deviation = q_back - initial_state
            v_component = torch.sum(deviation * V, dim=-1)
            
            # OTOC analog = squared deviation
            C_t = (v_component ** 2).item()
            otoc_values.append(C_t)
        
        return otoc_values


class ScramblingVisualizer:
    """Generate visualization data for scrambling diagnostics."""
    
    @staticmethod
    def format_result(result: ScramblingResult) -> Dict:
        """Format result for Neural Glass display."""
        return {
            "scrambling_time": f"{result.scrambling_time:.3f}",
            "lyapunov_exponent": f"{result.lyapunov_exponent:.4f}",
            "saturation_distance": f"{result.saturation_distance:.4f}",
            "is_chaotic": "✓ CHAOTIC" if result.is_chaotic else "○ Stable",
            "effective_beta": f"{result.effective_temperature:.2f}",
            "quantum_regime": "⚡ Above Hardness Threshold" if result.above_hardness_threshold else "◇ Classical Regime",
            "trajectory_length": len(result.distance_trajectory)
        }
    
    @staticmethod
    def create_plotly_trace(result: ScramblingResult) -> Dict:
        """Create Plotly trace data for distance trajectory."""
        t = np.arange(len(result.distance_trajectory)) * 0.05  # step_size
        return {
            "x": t.tolist(),
            "y": result.distance_trajectory,
            "type": "scatter",
            "mode": "lines",
            "name": "Geodesic Divergence",
            "line": {"color": "#00ffff", "width": 2}
        }
