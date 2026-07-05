"""
Proper Riemannian Manifold Geometry for IGBundle

This module implements true Riemannian geometric operations including:
- Riemannian metrics and geodesics
- Christoffel symbols and curvature tensors
- Parallel transport and covariant differentiation
- Exponential and logarithmic maps

Author: LLMOS SystemAgent
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class RiemannianMetric:
    """Represents a Riemannian metric tensor g_ij"""
    metric_tensor: torch.Tensor  # (B, T, P, D, D) - positive definite

    def __post_init__(self):
        # Ensure positive definiteness via Cholesky
        self.ensure_positive_definite()

    def ensure_positive_definite(self):
        """Ensure metric tensor is positive definite"""
        B, T, P, D, _ = self.metric_tensor.shape
        eps = 1e-6

        # Add identity to ensure positive definiteness
        eye = torch.eye(D, device=self.metric_tensor.device).expand_as(self.metric_tensor)
        self.metric_tensor = self.metric_tensor + eps * eye

        # Symmetrize
        self.metric_tensor = 0.5 * (self.metric_tensor + self.metric_tensor.transpose(-1, -2))

class RiemannianGeometry(nn.Module):
    """
    Implements Riemannian geometry operations on the bundle space.

    The base manifold B is equipped with a learned Riemannian metric,
    while fibers carry categorical structure with Fisher information metric.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim  # Dimension of base manifold
        self.num_components = config.num_components
        self.num_categories = config.num_categories

        # Learnable metric parameters (Cholesky factors)
        # g_ij = L * L^T where L is lower triangular
        # Stability: Initialize L close to Identity to start with Euclidean geometry
        # This prevents inverse metric explosion (singularities) at the start.
        self.metric_chol = nn.Parameter(
            torch.eye(self.dim).unsqueeze(0).repeat(self.num_components, 1, 1)
        )
        # Add small noise to break symmetry if needed, but Identity is safest start.
        self.metric_chol.data += torch.randn_like(self.metric_chol) * 0.01

        # Stability: Register hook to scale gradients of metric_chol
        # Because it is broadcast over B*T, gradients accumulate massively.
        # FIX: Reduced from 0.001 to 0.1 — 0.001 was starving the metric of gradient signal.
        self.metric_chol.register_hook(lambda grad: grad * 0.1)
        
        self.manifold_type = getattr(config, 'manifold_type', 'riemannian')

        # AUDIT FIX (2026-07): the conformal factor lambda(x) = 1 + a*tanh(||x||)
        # was hardcoded (a=0.5). Since log det g = D*log(lambda) + const, every
        # curvature statistic derived from it was invariant w.r.t. trained weights
        # (a constant of the architecture dressed as a measurement). With
        # config.learnable_conformal=True, `a` becomes trainable so the curvature
        # regularizer can actually bind. Default False preserves checkpoint
        # compatibility (non-persistent buffer => identical state_dict).
        if getattr(config, 'learnable_conformal', False):
            self.conformal_scale = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('conformal_scale', torch.tensor(0.5), persistent=False)

        if self.manifold_type == 'kan':
             # For KAN, we might need a different parameterized approach
             # But for now, we rely on the metric_chol parameters
             pass

    def _conformal_factor(self, positions: torch.Tensor) -> torch.Tensor:
        """lambda(x) = 1 + a * tanh(||x||), shape (..., 1).

        Single source of truth shared by get_metric and the curvature
        estimator (previously duplicated, risking silent divergence).
        """
        norm_x = torch.norm(positions, dim=-1, keepdim=True)
        return 1.0 + self.conformal_scale * torch.tanh(norm_x)

    def get_metric(self, positions: torch.Tensor) -> RiemannianMetric:
        """
        Compute Riemannian metric at given positions.

        Args:
            positions: (B, T, P, D) - Points on the manifold

        Returns:
            RiemannianMetric with tensor shape (B, T, P, D, D)
        """
        B, T, P, D = positions.shape
        
        # Safe Mode: Euclidean Manifold (Identity Metric)
        if self.manifold_type == 'euclidean':
            eye = torch.eye(D, device=positions.device)
            metric = eye.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, P, D, D)
            return RiemannianMetric(metric)

        # Extract Cholesky factor for each component
        L = torch.tril(self.metric_chol)  # (P, D, D) lower triangular

        # Compute metric as g = L * L^T
        L = torch.clamp(L, min=-5.0, max=5.0)
        
        metric = torch.matmul(L, L.transpose(-1, -2))  # (P, D, D)
        
        # Stability
        eye = torch.eye(D, device=metric.device).expand_as(metric)
        metric = metric + 1e-5 * eye

        # Broadcast to batch dimensions
        metric = metric.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)

        # Position-dependent conformal factor for non-zero curvature.
        # FIX: Increased from 0.1 to 0.5 — 0.1 capped curvature capacity at ~10%,
        # making it nearly impossible for the metric to express hyperbolic geometry.
        conformal_factor = self._conformal_factor(positions)  # (B, T, P, 1)
        conformal_factor = conformal_factor.unsqueeze(-1) # (B, T, P, 1, 1)
        
        metric = metric * conformal_factor

        return RiemannianMetric(metric)

    def christoffel_symbols(self, positions: torch.Tensor, metric: RiemannianMetric) -> torch.Tensor:
        """
        Compute Christoffel symbols.
        WARNING: This returns a (D, D, D) tensor which is O(D^3). 
        Use estimate_sectional_curvature_stochastic for large D.
        """
        # Return zeros (Euclidean approximation) to prevent OOM on legacy calls
        # Real curvature is handled by stochastic estimation which bypasses this.
        B, T, P, D = positions.shape
        # Only allocate if D is small
        if D > 64:
             # Implicitly zero (flat)
             return torch.zeros(B, T, P, 1, 1, 1, device=positions.device) # Dummy
        
        return torch.zeros(B, T, P, D, D, D, device=positions.device)

    @staticmethod
    def _safe_log_det(metric: torch.Tensor) -> torch.Tensor:
        """Numerically stable log|det(g)| using slogdet.

        torch.logdet backward overflows on large matrices (64x64 with det~1e10).
        slogdet decomposes into sign + log|det|, avoiding the overflow.
        We discard the sign since our metric is SPD (sign is always +1).
        """
        sign, logabsdet = torch.slogdet(metric)
        # Clamp to prevent extreme values from poisoning finite differences
        return logabsdet.clamp(-50.0, 50.0)

    def estimate_sectional_curvature_stochastic(self, positions: torch.Tensor,
                                              num_samples: int = 1) -> torch.Tensor:
        """
        Estimate sectional-scale curvature for the conformal metric g = lambda(x) * g0.

        AUDIT FIX (2026-07): the previous proxy returned -0.5 * Lap(log det g) / D,
        which dropped (i) the lambda^{-1} prefactor, (ii) the |grad log lambda|^2
        term, and (iii) the dimensional constants of the conformal scalar-curvature
        formula. This version implements the closed form for g = lambda * g0 with
        g0 position-constant (so R0 = 0):

            R = lambda^{-1} [ -(n-1) * Lap(log lambda)
                              - ((n-1)(n-2)/4) * |grad log lambda|^2 ]

        and returns the sectional scale kappa = R / (n(n-1)) so that a constant-
        curvature space of curvature K yields kappa = K.

        HONESTY CAVEATS (keep these when quoting telemetry):
        - Derivatives are taken w.r.t. Euclidean coordinates; exact only when
          g0 = L L^T ~ I (empirically true for trained checkpoints).
        - With config.learnable_conformal=False, lambda(x) is HARDCODED, so this
          value is weight-invariant: a constant of the architecture, NOT evidence
          of learned geometry. It must not be reported as a training outcome
          unless learnable_conformal=True.

        Args:
            positions: (B, T, P, D)
            num_samples: number of coordinate directions sampled for the
                stochastic Laplacian / gradient-norm estimates

        Returns:
            curvature: (B, T, P) - estimated sectional-scale curvature
        """
        B, T, P, D = positions.shape
        eps = 1e-3

        if self.manifold_type == 'euclidean' or D < 2:
            return torch.zeros(B, T, P, device=positions.device, dtype=positions.dtype)

        # f(x) = log lambda(x). The x-independent log det(L L^T) term cancels in
        # finite differences, so we work with lambda directly (single source of
        # truth via _conformal_factor).
        log_lam_base = torch.log(self._conformal_factor(positions)).squeeze(-1)  # (B, T, P)

        second_sum = torch.zeros_like(log_lam_base)
        grad_sq_sum = torch.zeros_like(log_lam_base)

        for _ in range(num_samples):
            k = torch.randint(0, D, (1,)).item()

            pos_plus = positions.clone()
            pos_plus[..., k] += eps
            log_lam_plus = torch.log(self._conformal_factor(pos_plus)).squeeze(-1)

            pos_minus = positions.clone()
            pos_minus[..., k] -= eps
            log_lam_minus = torch.log(self._conformal_factor(pos_minus)).squeeze(-1)

            # d^2 f / dx_k^2 and (df/dx_k)^2 along the sampled axis
            second_sum = second_sum + (log_lam_plus + log_lam_minus - 2.0 * log_lam_base) / (eps * eps)
            grad_sq_sum = grad_sq_sum + ((log_lam_plus - log_lam_minus) / (2.0 * eps)) ** 2

        # Stochastic trace estimators: E_k[d^2_k f] * D ~ Lap(f), E_k[(d_k f)^2] * D ~ |grad f|^2
        laplacian_log_lam = second_sum * (D / num_samples)
        grad_sq_log_lam = grad_sq_sum * (D / num_samples)

        lam = self._conformal_factor(positions).squeeze(-1)  # (B, T, P)

        # R = lambda^{-1} [ -(D-1) Lap(log lam) - ((D-1)(D-2)/4) |grad log lam|^2 ]
        scalar_R = (-(D - 1) * laplacian_log_lam
                    - ((D - 1) * (D - 2) / 4.0) * grad_sq_log_lam) / lam.clamp(min=1e-6)

        # Sectional scale: R = n(n-1) K for constant curvature K
        curvature = scalar_R / (D * (D - 1))

        return curvature

    def sectional_curvature(self, positions: torch.Tensor,
                            u: Optional[torch.Tensor] = None,
                            v: Optional[torch.Tensor] = None,
                            num_samples: int = 4) -> torch.Tensor:
        """Compatibility shim for legacy callers (bundle_curvature_loss,
        adaptive_curvature_loss).

        AUDIT FIX (2026-07): this method was removed when the neural-Christoffel
        machinery was retired, leaving callers to raise AttributeError. It now
        delegates to the stochastic conformal estimator. NOTE: the estimator
        returns a scalar-curvature-derived sectional scale; the direction
        arguments (u, v) are accepted for API compatibility but IGNORED — the
        conformal metric here is isotropic in the plane choice at this level of
        approximation.

        Returns:
            curvature: (B, T, P)
        """
        return self.estimate_sectional_curvature_stochastic(positions, num_samples=num_samples)

    def inner_product(self, u: torch.Tensor, v: torch.Tensor, metric: RiemannianMetric) -> torch.Tensor:
        """
        Compute Riemannian inner product g(u,v) = u^i g_{ij} v^j

        Args:
            u, v: (B, T, P, D) - vectors
            metric: RiemannianMetric

        Returns:
            inner_prod: (B, T, P) - scalar products
        """
        # g(u,v) = u^T G v
        return torch.einsum('...i,...ij,...j->...', u, metric.metric_tensor, v)

    def parallel_transport(self, vector: torch.Tensor, path: torch.Tensor,
                          metric: RiemannianMetric) -> torch.Tensor:
        """
        Parallel transport vector along path using Riemannian connection.

        Args:
            vector: (B, T, P, D) - initial vector at path[0]
            path: (B, T, P, N, D) - path points
            metric: RiemannianMetric

        Returns:
            transported: (B, T, P, D) - parallel transported vector
        """
        B, T, P, N, D = path.shape

        # Initialize with input vector
        current_vector = vector.clone()

        # Transport along path segments
        for i in range(N - 1):
            # Current position and next position
            pos_curr = path[..., i, :]
            pos_next = path[..., i + 1, :]

            # Compute Christoffel symbols at current position
            christoffel = self.christoffel_symbols(pos_curr, metric)

            # Path velocity
            velocity = pos_next - pos_curr

            # Parallel transport equation: DV/dt = -Γ^k_{ij} v^i γ^j V^k = 0
            # Discrete approximation: V_new ≈ V - Γ(γ, V) * dt
            connection_term = torch.einsum('...kij,...i,...j->...k',
                                         christoffel, velocity, current_vector)

            dt = 1.0 / (N - 1)  # Assuming unit parameter interval
            current_vector = current_vector - dt * connection_term

        return current_vector

    def exp_map(self, base_point: torch.Tensor, tangent_vec: torch.Tensor,
                metric: RiemannianMetric) -> torch.Tensor:
        """
        Riemannian exponential map: geodesic starting at base_point with initial velocity tangent_vec

        Args:
            base_point: (B, T, P, D) - starting point
            tangent_vec: (B, T, P, D) - initial velocity
            metric: RiemannianMetric

        Returns:
            endpoint: (B, T, P, D) - geodesic endpoint at parameter t=1
        """
        # Solve geodesic equation: d²γ/dt² + Γ^k_{ij} (dγ/dt)^i (dγ/dt)^j = 0
        # Using simple Euler integration (in practice, would use Runge-Kutta)

        num_steps = 10
        dt = 1.0 / num_steps

        position = base_point.clone()
        velocity = tangent_vec.clone()

        for step in range(num_steps):
            # Compute Christoffel symbols at current position
            christoffel = self.christoffel_symbols(position, metric)

            # Geodesic acceleration: -Γ^k_{ij} v^i v^j
            acceleration = -torch.einsum('...kij,...i,...j->...k', christoffel, velocity, velocity)

            # Update position and velocity
            position = position + dt * velocity
            velocity = velocity + dt * acceleration

        return position

    def log_map(self, base_point: torch.Tensor, target_point: torch.Tensor,
                metric: RiemannianMetric) -> torch.Tensor:
        """
        Riemannian logarithmic map: initial velocity of geodesic from base_point to target_point

        Args:
            base_point: (B, T, P, D)
            target_point: (B, T, P, D)
            metric: RiemannianMetric

        Returns:
            tangent_vec: (B, T, P, D) - logarithmic map
        """
        # AUDIT FIX (2026-07): christoffel_symbols returns zeros by design, so
        # exp_map integrates a straight line and log_map is exactly the Euclidean
        # difference. The LBFGS inner loop was an expensive no-op (and returned
        # detached, breaking gradients). Fast path taken whenever the connection
        # is flat; the LBFGS branch is retained for a future non-flat connection.
        christoffel_probe = self.christoffel_symbols(base_point, metric)
        if torch.count_nonzero(christoffel_probe) == 0:
            return target_point - base_point

        # Simplified implementation: use gradient descent to find initial velocity
        # such that exp_map(base_point, velocity) = target_point

        # Initialize with Euclidean difference
        initial_velocity = target_point - base_point
        velocity = nn.Parameter(initial_velocity.clone())

        optimizer = torch.optim.LBFGS([velocity], max_iter=20, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            predicted_endpoint = self.exp_map(base_point, velocity, metric)
            loss = F.mse_loss(predicted_endpoint, target_point)
            loss.backward()
            return loss

        optimizer.step(closure)

        return velocity.detach()

class FiberBundleLambdaCalculus(nn.Module):
    """
    Implements true fiber-to-fiber bundle lambda calculus operations.

    This provides:
    - Lambda abstraction over fiber bundle sections
    - Application of bundle morphisms
    - Categorical composition in fiber categories
    """

    def __init__(self, config):
        super().__init__()
        self.base_dim = config.latent_dim
        self.fiber_dim = config.num_categories
        self.num_components = config.num_components

        # Lambda abstraction network: encodes function types
        self.lambda_encoder = nn.Sequential(
            nn.Linear(self.base_dim + self.fiber_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.base_dim + self.fiber_dim)
        )

        # Application operator: applies function to argument
        self.application_net = nn.Sequential(
            nn.Linear(2 * (self.base_dim + self.fiber_dim), 256),
            nn.ReLU(),
            nn.Linear(256, self.base_dim + self.fiber_dim)
        )

        # Categorical composition in fibers
        self.fiber_compose = nn.Sequential(
            nn.Linear(2 * self.fiber_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.fiber_dim)
        )

    def lambda_abstraction(self, variable_type: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """
        Lambda abstraction: λx:A. body

        Args:
            variable_type: (B, T, P, D_base + D_fiber) - type of variable
            body: (B, T, P, D_base + D_fiber) - function body

        Returns:
            lambda_term: (B, T, P, D_base + D_fiber) - abstracted function
        """
        # Encode the lambda abstraction
        combined = torch.cat([variable_type, body], dim=-1)
        lambda_term = self.lambda_encoder(combined)
        return lambda_term

    def application(self, function: torch.Tensor, argument: torch.Tensor) -> torch.Tensor:
        """
        Function application: f @ x

        Args:
            function: (B, T, P, D_base + D_fiber) - lambda term
            argument: (B, T, P, D_base + D_fiber) - argument

        Returns:
            result: (B, T, P, D_base + D_fiber) - application result
        """
        # Combine function and argument
        combined = torch.cat([function, argument], dim=-1)
        result = self.application_net(combined)
        return result

    def fiber_morphism_compose(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Categorical composition of fiber morphisms: g ∘ f

        Args:
            f, g: (B, T, P, D_fiber) - fiber morphisms

        Returns:
            composed: (B, T, P, D_fiber) - g ∘ f
        """
        combined = torch.cat([f, g], dim=-1)
        composed = self.fiber_compose(combined)
        return composed

    def section_product(self, s1: torch.Tensor, s2: torch.Tensor,
                       base_coords: torch.Tensor) -> torch.Tensor:
        """
        Product of bundle sections over the base manifold

        Args:
            s1, s2: (B, T, P, D_base + D_fiber) - bundle sections
            base_coords: (B, T, P, D_base) - base manifold coordinates

        Returns:
            product: (B, T, P, D_base + D_fiber) - section product
        """
        # Split into base and fiber components
        s1_base, s1_fiber = s1[..., :self.base_dim], s1[..., self.base_dim:]
        s2_base, s2_fiber = s2[..., :self.base_dim], s2[..., self.base_dim:]

        # Base component: geometric mean weighted by base coordinates
        base_weight = F.softmax(base_coords, dim=-1)
        product_base = base_weight * s1_base + (1 - base_weight) * s2_base

        # Fiber component: categorical composition
        product_fiber = self.fiber_morphism_compose(s1_fiber, s2_fiber)

        return torch.cat([product_base, product_fiber], dim=-1)

def bundle_curvature_loss(geometry: RiemannianGeometry, positions: torch.Tensor,
                         target_curvature: float = -1.0) -> torch.Tensor:
    """
    Loss function to encourage specific curvature properties.

    AUDIT NOTE (2026-07): with config.learnable_conformal=False (default) the
    curvature estimate is a constant of the architecture (hardcoded lambda(x)),
    so this MSE optimizes a target no parameter can reach — the loss is
    unfalsifiable by construction. Enable learnable_conformal for it to bind.

    Args:
        geometry: RiemannianGeometry instance
        positions: (B, T, P, D) - points on manifold
        target_curvature: desired sectional curvature (negative for hyperbolic)

    Returns:
        loss: scalar - curvature regularization loss
    """
    B, T, P, D = positions.shape

    # Generate random orthonormal pairs of tangent vectors
    u = torch.randn_like(positions)
    v = torch.randn_like(positions)

    # Gram-Schmidt orthogonalization
    metric = geometry.get_metric(positions)
    u_norm = u / torch.sqrt(geometry.inner_product(u, u, metric) + 1e-8).unsqueeze(-1)

    u_v_inner = geometry.inner_product(u_norm, v, metric)
    v_orth = v - u_v_inner.unsqueeze(-1) * u_norm
    v_norm = v_orth / torch.sqrt(geometry.inner_product(v_orth, v_orth, metric) + 1e-8).unsqueeze(-1)

    # Compute sectional curvature
    sectional_k = geometry.sectional_curvature(positions, u_norm, v_norm)

    # L2 loss towards target curvature
    curvature_loss = F.mse_loss(sectional_k, torch.full_like(sectional_k, target_curvature))

    return curvature_loss