
import torch
import torch.nn as nn
from typing import Set, Dict, Optional, List
from .latent_store import FiberLatentStore

# Phase 2.5 Resonant Dynamics
try:
    from igbundle.dynamics.fhn import FHNIntegrator
except ImportError:
    FHNIntegrator = None

try:
    from igbundle.dynamics.eq_prop import EquilibriumPropagator
except ImportError:
    EquilibriumPropagator = None

    EquilibriumPropagator = None

try:
    from igbundle.dynamics.retrospection import TimeReversalCheck
except ImportError:
    TimeReversalCheck = None

class FiberExecutor:
    """
    Executes updates on the FiberLatentStore while enforcing 'Effect Discipline'.
    Only fibers in 'allowed_targets' are permitted to change significantly.
    Others are anchored to their previous state.
    """
    def __init__(self, store: FiberLatentStore):
        self.store = store
        
        # Resonant Core (Phase 2.5)
        self.resonator = None
        self.resonator_state = None
        
        if FHNIntegrator:
            # Assume num_fibers matches store size
            self.resonator = FHNIntegrator(store.n_fibers)
            self.resonator_state = (torch.zeros(store.n_fibers), torch.zeros(store.n_fibers))
            print(f"FiberExecutor: Resonant Dynamics Online (N={store.n_fibers})")
        else:
            print("FiberExecutor: Resonant Dynamics Unavailable.")
            
        # Physics-Based Learning (Phase 3)
        self.eq_propagator = None
        if self.resonator and EquilibriumPropagator:
            self.eq_propagator = EquilibriumPropagator(self.resonator, beta=0.5, lr=0.01)
            print("FiberExecutor: Equilibrium Propagation Online.")
            
        # Retrospection (Phase 3)
        self.retrospector = None
        if self.resonator and TimeReversalCheck:
            self.retrospector = TimeReversalCheck(self.resonator, tolerance=0.1)
            print("FiberExecutor: Time-Reversal Retrospection Online.")

    def update_resonance(self, active_indices: List[int]):
        """Step the FHN Resonator driven by active indices."""
        if not self.resonator: return
        
        # Construct Input Drive
        drive = torch.zeros(self.store.n_fibers)
        if active_indices:
             idx = torch.tensor(active_indices, dtype=torch.long)
             #drive[idx] = 1.0 # Pulse
             # Better: Add to existing drive? For now, instantaneous pulse.
             drive.scatter_(0, idx, 1.0)
        
        # Step Dynamics
        # Ensure we are not building a graph forever
        with torch.no_grad():
             q_new, p_new = self.resonator.step(self.resonator_state, drive)
             self.resonator_state = (q_new.detach(), p_new.detach())
        
        return q_new # Current excitation levels

    def learn_step(self, input_indices: List[int], target_indices: List[int]) -> float:
        """
        Perform one Equilibrium Propagation learning step.
        Adapts the Coupling Matrix J based on observed correlations.
        
        Args:
             input_indices: Fibers receiving external drive (Context).
             target_indices: Fibers that SHOULD be active (Prediction Target).
        """
        if not self.eq_propagator: return 0.0
        
        # Construct Patterns
        input_pattern = torch.zeros(self.store.n_fibers)
        if input_indices:
            idx = torch.tensor(input_indices, dtype=torch.long)
            input_pattern.scatter_(0, idx, 1.0)
            
        target_pattern = torch.zeros(self.store.n_fibers)
        if target_indices:
            t_idx = torch.tensor(target_indices, dtype=torch.long)
            target_pattern.scatter_(0, t_idx, 1.0) # Or should this be the FULL pattern?
            # Standard EqProp: Target is the "Correct Answer".
            # If target_indices is just the missing part, we should probably combine input + missing?
            # Let's assume target_indices represents the DESIRED state of the output nodes.
            # For auto-associative memory, target is the full pattern.
            
            # If we treat this as pattern completion:
            # Input: [1, 0, 0] (Partial)
            # Target: [1, 1, 0] (Full)
            # Just verify if target_indices covers the input too.
            # Let's assume the caller handles this logic.
        
        # Batch dimension required by Propagator
        input_batch = input_pattern.unsqueeze(0)
        target_batch = target_pattern.unsqueeze(0)
        
        loss = self.eq_propagator.learn_step(input_batch, target_batch)
        return loss

    def verify_thought(self, input_indices: List[int], enforce_jump: bool = False) -> tuple[float, bool]:
        """
        Phase 3: Verify if the current thought trajectory is reversible.
        If not, it implies information destruction (entropy generation) > tolerance.
        
        Returns:
            (loss, is_valid)
        """
        if not self.retrospector: return (0.0, True)
        
        # 1. Construct Drive
        drive = torch.zeros(self.store.n_fibers)
        if input_indices:
             idx = torch.tensor(input_indices, dtype=torch.long)
             drive.scatter_(0, idx, 1.0)
             
        # 2. Run Retrospection (Forward -> Reverse -> Backward)
        # We check if we can recover the start state from the result of the drive
        # Current state assumed to be "Start" (q0, p0)
        # We need to pass the CURRENT state as initial, or reset?
        # Argument: A valid thought is a trajectory from NOW.
        
        # We use a clone of current state to not disturb the actual mind
        q0, p0 = self.resonator_state
        state_snapshot = (q0.clone(), p0.clone())
        
        loss, valid = self.retrospector.retrospect(state_snapshot, drive)
        
        # If valid[0] is False, the thought is invalid.
        loss_val = loss.item() if loss.numel() == 1 else loss.mean().item()
        is_valid = valid.item() if valid.numel() == 1 else valid.all().item()
        
        if not is_valid:
            print(f"FiberExecutor: Retrospection Loss {loss_val:.4f}. Alert.")
            if enforce_jump:
                # Neurosymbolic Jump!
                # The thought was a dead end / irreversible collapse.
                print(f"FiberExecutor: Initiating Jump (Enforced).")
                self.hyper_jump(input_indices, intensity=1.5)
            
        return loss_val, is_valid

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
                
                # Resonant Gating (Phase 2.5)
                gate = 1.0
                if self.resonator and self.resonator_state:
                    q, _ = self.resonator_state
                    # Sigmoid gate based on fiber i's excitation
                    gate = torch.sigmoid(q[i]).item()
                
                # Store update (don't apply immediately to avoid order dependence)
                # Weighted by resonance!
                updates[i] = eta * mismatch * gate
                
            # Apply updates
            for i, delta in updates.items():
                self.store.s[i] += delta

            return {
                "off_locus_drift": off_locus.mean().item() if off_locus.numel() > 0 else 0.0,
                "on_locus_movement": on_locus.mean().item() if on_locus.numel() > 0 else 0.0,
                "closure_size": len(allowed_targets)
            }

    def hyper_jump(self, active_indices: Optional[List[int]] = None, intensity: float = 1.0, custom_direction: Optional[torch.Tensor] = None):
        """
        Force a Neurosymbolic Jump (Fiber Switch) to escape local optima.
        
        Mechanism:
        1. "Orthogonal Expansion": Project active fibers into the null space of their current meaning.
           s_new = Orthogonal(s_old)
            This effectively forces the system to find a 'perpendicular' thought.
        2. "Global Shake": Perturb the entire manifold configuration.
        """
        with torch.no_grad():
            # 1. Global Shake (Quantum Seed or Random)
            if custom_direction is not None:
                # Ensure device match
                if custom_direction.device != self.store.s.device:
                    custom_direction = custom_direction.to(self.store.s.device)
                
                # If custom direction provided (from Quantum Sampler), use it to shape the noise
                # We might need to resize if shape mismatch (e.g. sampler gives partial vector)
                if custom_direction.shape != self.store.s.shape:
                     # Resize or tile mechanism
                     flat_custom = custom_direction.flatten()
                     flat_target = self.store.s.flatten()
                     
                     if flat_custom.numel() == 0:
                         # Fallback to random if input is empty
                         rand_dir = torch.randn_like(self.store.s)
                     else:
                         # Cycle repeat
                         repeats = (flat_target.numel() // flat_custom.numel()) + 1
                         rand_dir = flat_custom.repeat(repeats)[:flat_target.numel()].reshape(self.store.s.shape)
                else:
                    rand_dir = custom_direction
            else:
                # Standard Random Noise
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
                
                s_new = (v_orth / v_orth_norm) * s_norm * intensity
                
                # Sanitize
                if torch.isnan(s_new).any():
                    print("FiberExecutor: NaNs detected in Hyper-Jump! Resetting to random.")
                    s_new = torch.randn_like(s_active)
                
                # Apply the Jump
                self.store.s[idx] = s_new
                
            # 3. Mild Global Perturbation for Context
            noise = torch.randn_like(self.store.s) * (intensity * 0.2)
            self.store.s.add_(noise)
            # Sanitize Store
            if torch.isnan(self.store.s).any():
                self.store.s.copy_(torch.randn_like(self.store.s))

            # 4. Reset Momentum
            if hasattr(self.store, 'p') and self.store.p is not None:
                self.store.p.zero_()
                
            return True
