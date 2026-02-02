import torch
import torch.nn as nn
import math

class PoincareBall(nn.Module):
    """
    Poincare Ball Model of Hyperbolic Geometry (K=-1).
    Implements Mobius Gysrovector Space operations.
    """
    def __init__(self, dim: int, c: float = 1.0):
        super().__init__()
        self.dim = dim
        self.c = c # Curvature parameter (c = -K). Standard is c=1.
        
    def _lambda(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor lambda_x = 2 / (1 - c||x||^2)"""
        x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        return 2.0 / (1.0 - self.c * x_sq).clamp(min=1e-5)

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        return self._lambda(x)
        
    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Mobius Addition: u (+) v
        """
        u2 = torch.sum(u * u, dim=-1, keepdim=True)
        v2 = torch.sum(v * v, dim=-1, keepdim=True)
        uv = torch.sum(u * v, dim=-1, keepdim=True)
        
        num = (1 + 2*self.c*uv + self.c*v2) * u + (1 - self.c*u2) * v
        den = 1 + 2*self.c*uv + self.c**2 * u2 * v2
        
        return num / den.clamp(min=1e-5)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at x: exp_x(v)
        = x (+) ( tanh( sqrt(c)*lambda_x*||v|| / 2 ) * v / (sqrt(c)*||v||) )
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
        lambda_x = self._lambda(x)
        
        sqrt_c = math.sqrt(self.c)
        factor = torch.tanh(sqrt_c * lambda_x * v_norm / 2) / (sqrt_c * v_norm)
        
        return self.mobius_add(x, factor * v)

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at x: log_x(y)
        """
        # (-x) (+) y
        sub = self.mobius_add(-x, y)
        sub_norm = torch.norm(sub, dim=-1, keepdim=True).clamp(min=1e-6)
        lambda_x = self._lambda(x)
        
        sqrt_c = math.sqrt(self.c)
        factor = (2 / (sqrt_c * lambda_x)) * torch.atanh(sqrt_c * sub_norm.clamp(max=1.0 - 1e-5))
        
        return factor * (sub / sub_norm)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic Distance d(x,y)
        """
        # (-x) (+) y
        sub = self.mobius_add(-x, y)
        sub_norm = torch.norm(sub, dim=-1, keepdim=True).clamp(min=1e-6)
        sqrt_c = math.sqrt(self.c)
        
        # d = (2/sqrt(c)) * atanh(sqrt(c) * ||-x + y||)
        dist = (2 / sqrt_c) * torch.atanh(sqrt_c * sub_norm.clamp(max=1.0 - 1e-5))
        return dist.squeeze(-1)

    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of v from x to y along geodesic.
        PT(x->y)(v) = lambda_x / lambda_y * gyr[y,-x](v)
        Usually simplified approximations work, or we implement Gyrovector rotation.
        For Phase 2 v1, let's use the scaling approximation which captures conformal scaling.
        But Gyration is crucial for holonomy.
        
        Simplified (Conformal Scaling Only):
        return v * (lambda_x / lambda_y) 
        
        (This is exact for vectors at origin, but loses rotation elsewhere. Sufficient for mild leapfrog?)
        Let's stick to scaling for v1 efficiency unless rigorous pt needed.
        """
        lambda_x = self._lambda(x)
        lambda_y = self._lambda(y)
        # return v * (lambda_x / lambda_y)
        # Wait, if we move from center to boundary, vector should shrink in Euclidean norm to represent same lengths?
        # Metric is lambda^2 I. 
        # Length^2 = lambda^2 ||v||^2.
        # Preserve Length: lambda_x ||v_x|| = lambda_y ||v_y|| => ||v_y|| = ||v_x|| * (lambda_x / lambda_y).
        return v * (lambda_x / lambda_y)
