
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

            return {
                "off_locus_drift": off_locus.mean().item() if off_locus.numel() > 0 else 0.0,
                "on_locus_movement": on_locus.mean().item() if on_locus.numel() > 0 else 0.0,
                "closure_size": len(allowed_targets)
            }

    def hyper_jump(self, active_indices: Optional[List[int]] = None, intensity: float = 1.0):
        """
        Force a Neurosymbolic Jump (Fiber Switch) to escape local optima.
        
        Mechanism:
        1. "Orthogonal Expansion": Project active fibers into the null space of their current meaning.
           s_new = Orthogonal(s_old)
            This effectively forces the system to find a 'perpendicular' thought.
        2. "Global Shake": Perturb the entire manifold configuration.
        """
        with torch.no_grad():
            # 1. Global Shake (Orthogonal Noise)
            # Create random direction
            rand_dir = torch.randn_like(self.store.s)
            
            # 2. Process Active Fibers (The "Stuck" Concepts)
            if active_indices:
                idx = torch.tensor(active_indices, device=self.store.s.device, dtype=torch.long)
                s_active = self.store.s[idx] # (k, D)
                
                # Compute Orthogonal Component relative to current state
                # v_orth = v - proj_u(v) = v - (v.u/u.u)u
                # Here u = s_active, v = random direction
                
                # Normalize active vectors for projection
                s_norm = torch.norm(s_active, dim=-1, keepdim=True) + 1e-6
                u = s_active / s_norm
                
                # Random vector v for each active fiber
                v = torch.randn_like(s_active)
                
                # Project v onto u: (v . u) * u
                dot = (v * u).sum(dim=-1, keepdim=True)
                proj = dot * u
                
                # Orthogonal component
                v_orth = v - proj
                
                # Renormalize to original energy level * intensity
                # This keeps the "importance" but changes the "meaning" to something independent.
                v_orth_norm = torch.norm(v_orth, dim=-1, keepdim=True) + 1e-6
                s_new = (v_orth / v_orth_norm) * s_norm * intensity
                
                # Apply the Jump
                self.store.s[idx] = s_new
                
            # 3. Mild Global Perturbation for Context
            noise = torch.randn_like(self.store.s) * (intensity * 0.2)
            self.store.s.add_(noise)

            # 4. Reset Momentum
            if hasattr(self.store, 'p') and self.store.p is not None:
                self.store.p.zero_()
                
            return True
