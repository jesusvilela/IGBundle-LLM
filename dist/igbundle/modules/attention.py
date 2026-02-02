import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PoincareAttention(nn.Module):
    """
    Implements Geodesic Attention on the Poincaré Ball.
    
    Attention weights are derived from hyperbolic distances:
    A_ij = softmax(- distance(q_i, k_j) / temperature)
    
    This enforces a "Hierarchical Bias": tokens attend more strongly to 
    semantically/geometrically close concepts (neighbors in the tree)
    rather than just dot-product similarity.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1)) # Learnable temp
        
        # Linear projections for Value (mixing information)
        # Query/Key are the geometric coordinates themselves!
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _poincare_distance_squared(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes squared Poincaré distance between x and y.
        d(u,v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
        
        Returns the argument of arccosh (delta) which is monotonic with distance.
        Using delta directly in attention is numerically stable and sufficient for ranking.
        """
        # x, y: (B, H, T, D_head)
        
        # Euclidean norms
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        
        # Clamp norms to avoid boundary division by zero
        x_norm_sq = torch.clamp(x_norm_sq, max=0.99)
        y_norm_sq = torch.clamp(y_norm_sq, max=0.99)
        
        # Euclidean distance squared
        dist_sq = torch.sum((x - y) ** 2, dim=-1)
        
        # Mobius scaling
        scale = (1 - x_norm_sq) * (1 - y_norm_sq) # (B, H, T, 1) * (B, H, S, 1) -> broadcasting needed
        
        # We need pairwise distances: (T, T)
        # Let's reshape for broadcasting:
        # x: (B, H, T, 1, D)
        # y: (B, H, 1, T, D)
        
        return dist_sq # Placeholder for full pairwise, implementing in forward
        
    def _pairwise_poincare_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full pairwise distance matrix for self-attention.
        x: (B, P, T, D) -> P heads usually correspond to components
        """
        B, T, D = x.shape
        
        # Expand for pairwise
        x_i = x.unsqueeze(2) # (B, T, 1, D)
        x_j = x.unsqueeze(1) # (B, 1, T, D)
        
        # Squared Euclidean distance
        euclidean_d2 = torch.sum((x_i - x_j)**2, dim=-1) # (B, T, T)
        
        # Norms
        x_norm2 = torch.sum(x**2, dim=-1) # (B, T)
        x_norm2 = torch.clamp(x_norm2, max=0.99)
        
        alpha = 1 - x_norm2.unsqueeze(2) # (B, T, 1)
        beta = 1 - x_norm2.unsqueeze(1)  # (B, 1, T)
        
        # Poincaré delta
        delta = 1 + 2 * euclidean_d2 / (alpha * beta + 1e-8)
        
        # Distance
        dist = torch.acosh(torch.clamp(delta, min=1.0 + 1e-6))
        return dist

    def _log_map_zero(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from manifold origin to tangent space (T_0 D)."""
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-15, max=0.99999)
        scale = torch.atanh(x_norm) / x_norm
        return x * scale

    def _exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space origin to manifold (D)."""
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-15)
        # Tanh compression ensures we stay inside unit ball regardless of vector magnitude
        scale = torch.tanh(v_norm) / v_norm
        return v * scale

    def forward(self, coords: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, T, P, D) - Geometric coordinates (acting as Q and K)
            values: (B, T, H) or (B, T, D) - Features to transport
        """
        B, T, P, D = coords.shape
        
        # We treat P (components) effectively as Heads for attention?
        # Or we apply attention independently per component?
        # Let's apply per-component attention for now.
        
        # Reshape coords to (B*P, T, D) to batch components
        coords_flat = coords.view(B*P, T, D)
        
        # Compute Geodesic Distance Matrix
        # (B*P, T, T)
        dist_matrix = self._pairwise_poincare_dist(coords_flat)
        
        # Attention Logits: -dist / temp
        attn_logits = -dist_matrix / torch.clamp(self.temperature, min=0.01)
        
        # Causal Mask (if needed - usually adapters preserve causality)
        # mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(coords.device)
        # attn_logits.masked_fill_(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Mix Values using Tangent Space Aggregation (Einstein Midpoint approximation)
        # 1. Map values to Tangent Space at origin
        # Assuming values are points on the manifold (same as coords)
        
        # Note: If values are NOT on manifold (e.g. Euclidean vectors), we skip log_map.
        # But here valid input 'values' = 'coords' (for now).
        # v_flat = values.view(B*P, T, D)
        
        # Let's project coords into a Value space via MLP *then* map to tangent?
        # Or just use coords directly. Let's use coords directly as "Pure Geometric Attention".
        v_flat = coords_flat 
        
        # Map to Tangent Space T_0
        v_tangent = self._log_map_zero(v_flat) # (B*P, T, D)
        
        # 2. Weighted Sum in Tangent Space (Euclidean operation valid here)
        out_tangent = torch.bmm(attn_weights, v_tangent) # (B*P, T, D)
        
        # 3. Map back to Manifold via Exp Map
        out_manifold = self._exp_map_zero(out_tangent)
        
        # Reshape back
        out = out_manifold.view(B, T, P, D)
        out = self.out_proj(out)
        
        return out
