
import torch
import torch.nn as nn
from typing import Set, Dict, Optional, List
from .latent_store import FiberLatentStore

class FiberExecutor:
    """
    Executes updates on the FiberLatentStore while enforcing 'Effect Discipline'.
    Only fibers in 'allowed_targets' are permitted to change significantly.
    Others are anchored to their previous state.
    """
    def __init__(self, store: FiberLatentStore):
        self.store = store

    def anchor_off_locus(self, allowed_targets: Set[int], beta: float = 0.1):
        """
        Soft-reset off-locus fibers towards their snapshot state.
        s_off = (1-beta)*s_off + beta*s_prev_off
        """
        if self.store.s_prev is None:
            return

        with torch.no_grad():
            # Create boolean mask for ALL fibers
            mask = torch.ones(self.store.n_fibers, dtype=torch.bool, device=self.store.s.device)
            
            # Unmask allowed targets (they are free to move)
            if allowed_targets:
                indices = torch.tensor(list(allowed_targets), device=self.store.s.device, dtype=torch.long)
                mask[indices] = False
                
            # Apply anchor to masked (off-locus) fibers
            # Note: We assume optimizer might have moved them slightly; this pulls them back.
            # Or if we update manually, this enforces stability.
            current_vals = self.store.s[mask]
            prev_vals = self.store.s_prev[mask]
            
            # Linear interpolation towards previous state
            self.store.s[mask] = (1.0 - beta) * current_vals + beta * prev_vals

    def propagate(self, adjacency: Dict[int, Set[int]], allowed_targets: Set[int], eta: float = 0.05):
        """
        Simple smoothing propagation along edges.
        s_i += eta * mean(s_j - s_i) for j in neighbors
        Only applies to allowed_targets to strictly respect discipline.
        """
        if not allowed_targets:
            return

        with torch.no_grad():
            updates = {}
            
            for i in allowed_targets:
                neighbors = adjacency.get(i, set())
                if not neighbors:
                    continue
                    
                # Get neighbor latents
                nbr_indices = torch.tensor(list(neighbors), device=self.store.s.device, dtype=torch.long)
                s_neighbors = self.store.s[nbr_indices] # (num_nbrs, d_s)
                s_i = self.store.s[i] # (d_s)
                
                # Compute effective mismatch (mean difference)
                # primitive "Laplacian smoothing"
                mismatch = s_neighbors.mean(dim=0) - s_i
                
                # Store update (don't apply immediately to avoid order dependence)
                updates[i] = eta * mismatch
                
            # Apply updates
            for i, delta in updates.items():
                self.store.s[i] += delta

    def compute_drift_metrics(self, allowed_targets: Set[int]) -> Dict[str, float]:
        """Compute drift statistics for logging."""
        if self.store.s_prev is None:
            return {"off_locus_drift": 0.0, "on_locus_movement": 0.0}
            
        with torch.no_grad():
            diff = self.store.s - self.store.s_prev
            dist = torch.norm(diff, dim=-1) # (N,)
            
            mask = torch.ones(self.store.n_fibers, dtype=torch.bool, device=self.store.s.device)
            if allowed_targets:
                 indices = torch.tensor(list(allowed_targets), device=self.store.s.device, dtype=torch.long)
                 mask[indices] = False
                 
            off_locus = dist[mask]
            on_locus = dist[~mask]
            
            return {
                "off_locus_drift": off_locus.mean().item() if off_locus.numel() > 0 else 0.0,
                "on_locus_movement": on_locus.mean().item() if on_locus.numel() > 0 else 0.0,
                "closure_size": len(allowed_targets)
            }
