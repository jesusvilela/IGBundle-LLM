"""
Riemannian Attention Operator
-----------------------------
Implements the geometric attention mechanism for Phase 7.
Replaces Dot-Product Attention (Euclidean) with Geodesic Distance Scoring.

References:
- Gulcehre et al., "Hyperbolic Attention Networks" (ICLR 2019)
- Hypformer (NeurIPS 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..geometry.hyperbolic import PoincareBall

class RiemannianAttention(nn.Module):
    def __init__(self, config, cache_manager=None):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.cache_manager = cache_manager
        
        # Curvature c
        self.c = getattr(config, 'manifold_curvature', 1.0)
        self.tau = getattr(config, 'geometric_temperature', 1.0) # Temperature for softmax
        self.use_cache = getattr(config, 'use_geometric_cache', False)
        self.use_nmtrixed = getattr(config, 'use_nmtrixed', True)
        
        # Learnable temperature?
        self.learn_tau = getattr(config, 'learn_geometric_tau', False)
        if self.learn_tau:
            self.tau_param = nn.Parameter(torch.tensor(self.tau))
            
    def _nmtrix(self, x: torch.Tensor) -> torch.Tensor:
        """Return a normalized manifold kernel representation."""
        return PoincareBall.kernelized_coords(x, self.c)

    def forward(self, 
                query_states: torch.Tensor, # (B, H, T, D) - Query on Manifold (projected)
                key_states: torch.Tensor,   # (B, H, S, D) - Keys on Manifold
                value_states: torch.Tensor, # (B, H, S, D) - Values (Typically Euclidean or Tangent Space)
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Riemannian Attention.
        
        Score(q, k) = - d_M(q, k)^2 / tau
        """
        if self.use_cache and self.cache_manager is not None:
            self.cache_manager.update(0, key_states, value_states, projector=None)

        q_states = query_states
        k_states = key_states
        if self.use_nmtrixed:
            q_states = self._nmtrix(query_states)
            k_states = self._nmtrix(key_states)

        dist_sq = PoincareBall.pairwise_dist2_fast(q_states, k_states, self.c)

        tau = self.tau_param if self.learn_tau else self.tau
        scores = -dist_sq / tau

        if attention_mask is not None:
            scores = scores + attention_mask
             
        attn_weights = F.softmax(scores, dim=-1) # (B, H, T, S)
        
        context_layer = torch.matmul(attn_weights, value_states)
        return context_layer
