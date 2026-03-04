"""
Poincaré Ball Geometric Kernel
------------------------------
Optimized, numerically stable implementations of hyperbolic geometry operations
for the Riemannian KV Cache and Adapter.

References:
- Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018)
- Skopek et al., "Mixed-curvature Variational Autoencoders" (ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Numerical stability constants
# Numerical stability constants
EPS = 1e-3
MIN_NORM = 1e-5
MAX_TANH_ARG = 10.0

class PoincareBall:
    """
    Static kernel for Poincaré Ball operations with curvature c.
    Note: c > 0 implies curvature K = -c.
    """
    
    @staticmethod
    def _clamp_tanh(x: torch.Tensor) -> torch.Tensor:
        """Safe tanh to avoid NaN during backward pass."""
        return torch.tanh(torch.clamp(x, min=-MAX_TANH_ARG, max=MAX_TANH_ARG))

    @staticmethod
    def _clamp_artanh(x: torch.Tensor) -> torch.Tensor:
        """Safe artanh to avoid NaN."""
        x = torch.clamp(x, min=-1.0 + EPS, max=1.0 - EPS)
        return torch.atanh(x)

    @staticmethod
    def lambda_x(x: torch.Tensor, c: float, keepdim: bool = False) -> torch.Tensor:
        """
        Compute conformal factor lambda_x = 2 / (1 - c * ||x||^2).
        """
        x_sqnorm = torch.sum(x.pow(2), dim=-1, keepdim=keepdim)
        return 2.0 / (1.0 - c * x_sqnorm).clamp_min(MIN_NORM)

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """
        Möbius Addition: x (+) y
        
        Formula:
        (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y
        -------------------------------------------
        1 + 2c<x,y> + c^2||x||^2||y||^2
        """
        x2 = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y2 = torch.sum(y.pow(2), dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c**2 * x2 * y2
        
        return num / denom.clamp_min(MIN_NORM)

    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """
        Geodesic Distance: d(x, y) = 1/sqrt(c) * arccosh(1 + ...)
        
        Formula:
        1 + 2c ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))
        """
        sqrt_c = c ** 0.5
        x2 = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y2 = torch.sum(y.pow(2), dim=-1, keepdim=True)
        dist_sq = torch.sum((x - y).pow(2), dim=-1, keepdim=True)
        
        alpha = 1.0 - c * x2
        beta = 1.0 - c * y2
        
        # Stability: clamp denominators
        alpha = alpha.clamp_min(MIN_NORM)
        beta = beta.clamp_min(MIN_NORM)
        
        num = 2.0 * c * dist_sq
        denom = alpha * beta
        
        arg = 1.0 + num / denom

        # Numerically stable arcosh.
        # arcosh(z) = log(z + sqrt(z^2 - 1)) has NaN gradient when z~1
        # because sqrt(0) has infinite backward (1/(2*sqrt(0))).
        # Fix: clamp (z-1) away from 0 before sqrt, giving a safe minimum distance.
        arg = arg.clamp_min(1.0 + 1e-6)
        dist = torch.acosh(arg)

        return dist / sqrt_c

    @staticmethod
    def dist2_fast(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """
        Squared Distance (Faster, for scoring).
        Returns d(x,y)^2.
        """
        # Optimization: Don't take sqrt then square?
        # dist = 1/sqrt(c) * acosh(...)
        # dist^2 = 1/c * acosh(...)^2
        # We still need acosh.
        d = PoincareBall.dist(x, y, c)
        return d.pow(2)

    @staticmethod
    def exp_map(x: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
        """
        Exponential Map: Map tangent vector v at x to manifold.
        Exp_x(v) = x (+) (tanh(sqrt(c)/2 * lambda_x * ||v||) * v / (sqrt(c)*||v||))
        """
        sqrt_c = c ** 0.5
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        
        lambda_x = PoincareBall.lambda_x(x, c, keepdim=True)
        
        # Use float64 for intermediate precision if inputs are standard
        # But we must respect input dtype. 
        
        scale = PoincareBall._clamp_tanh(sqrt_c * lambda_x * norm_v / 2.0)
        scale = scale / (sqrt_c * norm_v)
        
        u = scale * v
        
        # If x is near zero, Exp_0(v) ~ tanh(sqrt(c)|v|)*v/(...)
        # But we use generic Mobius Add for transport stability
        # Check if x is close to zero for optimization?
        if torch.sum(x.pow(2)) < MIN_NORM:
             # Exp_0(v) = tanh(sqrt(c)|v|) * v / (sqrt(c)|v|)
             return PoincareBall._clamp_tanh(sqrt_c * norm_v) * v / (sqrt_c * norm_v)

        return PoincareBall.mobius_add(x, u, c)

    @staticmethod
    def log_map(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """
        Logarithmic Map: Map y on manifold to tangent space of x.
        Log_x(y) = 2/ (sqrt(c)*lambda_x) * artanh(sqrt(c) * || -x (+) y ||) * ...
        """
        sqrt_c = c ** 0.5
        sub = PoincareBall.mobius_add(-x, y, c)
        norm_sub = torch.norm(sub, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        
        lambda_x = PoincareBall.lambda_x(x, c, keepdim=True)
        
        term = PoincareBall._clamp_artanh(sqrt_c * norm_sub)
        scale = 2.0 / (sqrt_c * lambda_x) * (term / norm_sub)
        
        return scale * sub

    @staticmethod
    def project(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
        """
        Project x back into the Poincaré ball if it strays outside.
        ||x|| < 1/sqrt(c)
        """
        max_norm = (1.0 - eps) / (c ** 0.5)
        norm = torch.norm(x, dim=-1, keepdim=True)
        
        cond = norm > max_norm
        projected = x / norm * max_norm
        
        return torch.where(cond, projected, x)
    
    @staticmethod
    def from_euclidean(x: torch.Tensor, c: float) -> torch.Tensor:
        """
        Map Euclidean R^n to Poincaré Ball via exponential map at 0.
        Commonly: tanh(sqrt(c)||x||) * x/||x||
        """
        sqrt_c = c ** 0.5
        norm_x = torch.norm(x, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        
        # Use tanh to squash R^n into ball
        target_norm = PoincareBall._clamp_tanh(sqrt_c * norm_x)
        return (target_norm / (sqrt_c * norm_x)) * x

    @staticmethod
    def parallel_transport(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
        """
        Parallel Transport of v from tangent space T_x to T_y along geodesic.
        PT_{x->y}(v) = lambda_x/lambda_y * gyr[y, -x] v
        
        Approximation: For adaptation, we often just use Gyration.
        Exact formula involves mobius addition commutativity scaling.
        
        Ref: Ganea 2018 eq 4.
        """
        # Gyration term
        # gyr[u, v] w = -(u+v) + (u + (v+w))
        # This is expensive.
        
        # Approximated Transport for speed:
        # Just scale by conformal factor ratio?
        # lambda_x = PoincareBall.lambda_x(x, c, True)
        # lambda_y = PoincareBall.lambda_x(y, c, True)
        # return v * (lambda_x / lambda_y)
        
        # Actually, let's implement Gyration for Phase 7 correctness.
        # gyr[y, -x] v
        # We need Mobius addition.
        
        # u = -x
        # v_ = y
        # We compute gyr[v_, u] v
        
        u = -x
        v_ = y
        
        # a = u + v_
        a = PoincareBall.mobius_add(u, v_, c)
        
        # b = v_ + v_vec (Wait, gyration acts on vector?)
        # Interpretation: gyration rotates the vector.
        # Formula: gyr[u, v] w = ...
        
        # Simplification for KV Cache Phase 7:
        # We might not need Full Parallel Transport if we use LogMap aggregation at 0.
        # If we aggregate at 0, transport is just P_x->0.
        # P_x->0 (v) = (1 - c||x||^2) / (1 + c||x||^2) * ... ??? Usually just scaling.
        
        # Let's start with NO-OP or Scaling, and add Gyration later if direction matters.
        lambda_x = PoincareBall.lambda_x(x, c, True)
        lambda_y = PoincareBall.lambda_x(y, c, True)
        return v * (lambda_x / lambda_y)

class ManifoldTensor(nn.Module):
    """
    Wrapper for tensors living on the manifold.
    Enforces constraints and provides utility methods.
    """
    def __init__(self, data: torch.Tensor, c: float = 1.0):
        super().__init__()
        self.data_tensor = nn.Parameter(data)
        self.c = c
        self.register_buffer('curvature', torch.tensor(c))
        
    def forward(self):
        return self.data_tensor
        
    def project_(self):
        """In-place projection to ensure validity."""
        with torch.no_grad():
            self.data_tensor.data = PoincareBall.project(self.data_tensor.data, self.c)
