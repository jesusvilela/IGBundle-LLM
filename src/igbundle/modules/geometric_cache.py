"""
Geometric KV Cache
------------------
Implements a dual-memory KV Cache that stores:
1. Standard Key/Value tensors (Euclidean) for the frozen backbone.
2. Geometric Key projectons (Manifold) for Riemannian attention.

This module manages the storage, retrieval, and eviction policies based on curvature.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from ..geometry.hyperbolic import PoincareBall

@dataclass
class GeometricCacheState:
    """State for a single layer's cache."""
    key_cache_euclidean: Optional[torch.Tensor] = None # (B, H, S, D)
    value_cache_euclidean: Optional[torch.Tensor] = None # (B, H, S, D)
    key_cache_manifold: Optional[torch.Tensor] = None # (B, H, S, D) - On Manifold
    # We don't necessarily cache manifold values if we aggregate in tangent space of 0
    # But if we did Einstein Midpoint, we might need value_cache_manifold or just reuse euclidean values mapped?
    # For now, let's assume we map values on the fly or reuse euclidean V.
    
    curvature_scores: Optional[torch.Tensor] = None # (B, S) - Importance score for eviction

class GeometricKVCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_length = getattr(config, 'max_position_embeddings', 32768)
        self.curvature_c = getattr(config, 'manifold_curvature', 1.0)
        self.eviction_policy = getattr(config, 'geometric_eviction', False)
        
    def update(self, 
               key_states: torch.Tensor, 
               value_states: torch.Tensor, 
               layer_idx: int, 
               cache_kwargs: Optional[Dict] = None,
               geometric_projector: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new tokens.
        
        Args:
            key_states: (B, H, NewTokens, D) - Standard Keys
            value_states: (B, H, NewTokens, D) - Standard Values
            layer_idx: Layer index
            cache_kwargs: dict containing 'sin', 'cos', etc.
            geometric_projector: Optional module to map K -> K_manifold
            
        Returns:
            Tuple of (keys, values) to be used in attention.
            Note: Returns Euclidean K/V for standard attention if this module is used as a wrapper.
            To use Riemannian attention, the caller must access .manifold_cache directly.
        """
        # Update is handled by the specific attention implementation usually.
        # But here we provide the storage logic.
        
        # This function signature mimics transformers 4.x DynamicCache
        # But we need to handle the stateful nature differently or subclass DynamicCache.
        pass

    # We'll implement a functional update for now that takes previous cache and returns new.
    # Because 'transformers' cache API is complex to subclass perfectly without inheritance.
    
    @staticmethod
    def update_layer(prev_state: GeometricCacheState,
                     new_k: torch.Tensor,
                     new_v: torch.Tensor,
                     geometric_projector: Optional[nn.Module] = None,
                     c: float = 1.0) -> GeometricCacheState:
        """
        Functional update for a single layer.
        """
        # Update Euclidean
        if prev_state.key_cache_euclidean is None:
            k_eucl = new_k
            v_eucl = new_v
        else:
            k_eucl = torch.cat([prev_state.key_cache_euclidean, new_k], dim=2)
            v_eucl = torch.cat([prev_state.value_cache_euclidean, new_v], dim=2)
            
        # Update Manifold (if projector provided)
        k_mani = prev_state.key_cache_manifold
        if geometric_projector is not None:
            # We only project the NEW keys to save compute!
            # new_k shape: (B, H, L_new, D)
            # Projector expects (..., D)
            new_k_mani = geometric_projector(new_k) # Should return points on manifold
            
            if k_mani is None:
                k_mani = new_k_mani
            else:
                k_mani = torch.cat([k_mani, new_k_mani], dim=2)
        
        return GeometricCacheState(
            key_cache_euclidean=k_eucl,
            value_cache_euclidean=v_eucl,
            key_cache_manifold=k_mani,
            curvature_scores=None # TODO: Update scores
        )
    
    @staticmethod
    def evict_flat_regions(state: GeometricCacheState, 
                           keep_ratio: float = 0.8,
                           window_size: int = 10) -> GeometricCacheState:
        """
        Eviction Policy: Remove tokens that are in 'flat' regions of the manifold
        (low curvature/importance), keeping 'pivots'.
        
        Note: This is complex because removing from KV breaks the sequence position indices
        unless we use attention masks or rotary embedding adjustments.
        For Phase 7 MVP, we might skip this or just mark them.
        """
        # Placeholder for Epic 35 extensions
        return state

class GeometricCacheManager(nn.Module):
    """
    Manages cache for all layers.
    Compatible with Hugging Face 'past_key_values' logic if wrapped.
    """
    def __init__(self, config):
        super().__init__()
        self.caches: List[GeometricCacheState] = [GeometricCacheState() for _ in range(config.num_hidden_layers)]
        self.config = config
        
    def get_layer(self, layer_idx: int) -> GeometricCacheState:
        return self.caches[layer_idx]
        
    def update(self, layer_idx, new_k, new_v, projector=None):
        self.caches[layer_idx] = GeometricKVCache.update_layer(
            self.caches[layer_idx], new_k, new_v, projector, getattr(self.config, 'manifold_curvature', 1.0)
        )
        return self.caches[layer_idx]
