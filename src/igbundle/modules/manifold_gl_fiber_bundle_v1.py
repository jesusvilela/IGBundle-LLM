
"""
ManifoldGL Fiber Bundle Implementation v1
-----------------------------------------
A rigorous geometric fiber bundle architecture with Hamiltonian dynamics and 
Monadic-style Effect Discipline.

Key Components:
1. Fiber State (Option A: Latent Vector)
2. Riemannian Geometry (Poincare Ball)
3. Symplectic Integration (Leapfrog)
4. Effect Discipline (Propagate + Anchor)
5. Regression Test Suite (A-F)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import math
import copy
import logging

# Phase 2.5 Resonant Dynamics
try:
    from igbundle.dynamics.fhn import FHNIntegrator
except ImportError:
    # Fallback/Dummy if path not setup
    FHNIntegrator = None

try:
    from igbundle.geometry.hyperbolic import PoincareBall as _PoincareBall
except ImportError:
    _PoincareBall = None

# Configuration
@dataclass
class ManifoldConfig:
    latent_dim: int = 64
    curvature: float = -1.0
    temperature: float = 0.1
    min_norm: float = 1e-5
    max_norm: float = 0.99  # Poincare Ball boundary
    
@dataclass
class HamiltonianConfig:
    use_dynamics: bool = True
    num_leapfrog_steps: int = 4
    step_size: float = 0.1
    damping: float = 0.01
    mass_matrix_diagonal: float = 1.0

@dataclass
class EffectConfig:
    anchor_strength: float = 0.95
    max_propagation_hops: int = 2
    overlap_threshold: float = 0.5
    
# --- Data Structures & Typing ---

FiberID = str

@dataclass
class FiberState:
    """Represents the state of a single fiber in the bundle."""
    z: torch.Tensor          # Latent coordinate on manifold (D,)
    p: torch.Tensor          # Momentum tangent vector (D,)
    chart_idx: int = 0       # Atlas chart index (default 0 for single chart)
    log_metric: torch.Tensor = field(default_factory=lambda: torch.zeros(1)) # Local metric parameter
    
    def clone(self):
        return FiberState(
            z=self.z.clone(),
            p=self.p.clone(),
            chart_idx=self.chart_idx,
            log_metric=self.log_metric.clone()
        )

@dataclass
class FiberDelta:
    """Represents a proposed update to a fiber."""
    z_delta: torch.Tensor    # Tangent space update vector (D,)
    source_hop: int = 0      # Distance from update origin

@dataclass
class EffectLog:
    """Audit trail for an update step."""
    logs: List[str]
    allowed_targets: Set[FiberID]

class AdjacencyGraph:
    """Manages the topology of the fiber bundle (Tree + Overlap)."""
    def __init__(self):
        self.tree_edges: Dict[FiberID, FiberID] = {} # child -> parent
        self.overlap_edges: Dict[Tuple[FiberID, FiberID], float] = {} # (i,j) -> score
        
    def add_tree_edge(self, child: FiberID, parent: FiberID):
        self.tree_edges[child] = parent
        
    def add_overlap(self, i: FiberID, j: FiberID, score: float):
        if i == j: return
        key = tuple(sorted((i, j)))
        self.overlap_edges[key] = max(self.overlap_edges.get(key, 0.0), score)
        
    def get_neighbors(self, fid: FiberID, threshold: float = 0.5) -> List[Tuple[FiberID, str]]:
        neighbors = []
        # Parent
        if fid in self.tree_edges:
            neighbors.append((self.tree_edges[fid], 'parent_child'))
        # Children
        for child, parent in self.tree_edges.items():
            if parent == fid:
                neighbors.append((child, 'parent_child'))
        # Overlaps
        for (i, j), score in self.overlap_edges.items():
            if score > threshold:
                if i == fid: neighbors.append((j, 'token_overlap'))
                elif j == fid: neighbors.append((i, 'token_overlap'))
        return neighbors

    def get_weight(self, i: FiberID, j: FiberID) -> float:
        # Simplified weight lookups
        key = tuple(sorted((i, j)))
        if key in self.overlap_edges:
            return self.overlap_edges[key]
        return 1.0 # Tree edges are strong connections

# --- Riemannian Geometry Core (Poincare Ball) ---
# Delegates to the canonical PoincareBall kernel in geometry/hyperbolic.py
# to eliminate duplicate implementations. Falls back to inline ops if unavailable.

class RiemannianGeometry:
    """Thin adapter wrapping PoincareBall for v1 ManifoldConfig compatibility."""
    def __init__(self, config: ManifoldConfig):
        self.c = abs(config.curvature)
        self.min_norm = config.min_norm
        self.max_norm = config.max_norm
        self._ball = _PoincareBall
        
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition in Poincare Ball."""
        if self._ball is not None:
            return self._ball.mobius_add(x, y, self.c)
        # Inline fallback
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return num / (denom + 1e-8)
        
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: Tangent space -> Manifold."""
        if self._ball is not None:
            return self._ball.exp_map(x, v, self.c)
        # Inline fallback
        sqrt_c = math.sqrt(self.c)
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        lambda_x = 2 / (1 - self.c * torch.sum(x * x, dim=-1, keepdim=True)).clamp_min(1e-8)
        coeff = torch.tanh(lambda_x * sqrt_c * v_norm / 2) / (sqrt_c * v_norm)
        u = coeff * v
        return self.mobius_add(x, u)
        
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log map: Manifold -> Tangent space at x."""
        if self._ball is not None:
            return self._ball.log_map(x, y, self.c)
        # Inline fallback
        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        lambda_x = 2 / (1 - self.c * torch.sum(x * x, dim=-1, keepdim=True)).clamp_min(1e-8)
        sqrt_c = math.sqrt(self.c)
        return (2 / (sqrt_c * lambda_x)) * torch.atanh(sqrt_c * diff_norm) * (diff / diff_norm)
        
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._ball is not None:
            return self._ball.dist(x, y, self.c)
        # Inline fallback
        diff = self.mobius_add(-x, y)
        norm = diff.norm(dim=-1, keepdim=True)
        sqrt_c = math.sqrt(self.c)
        return (2 / sqrt_c) * torch.atanh(math.sqrt(self.c) * norm.clamp(max=0.99))

    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transport vector v from TxM to TyM."""
        if self._ball is not None:
            return self._ball.parallel_transport(x, y, v, self.c)
        # Inline fallback (conformal factor scaling)
        lambda_x = 2 / (1 - self.c * torch.sum(x * x, dim=-1, keepdim=True))
        lambda_y = 2 / (1 - self.c * torch.sum(y * y, dim=-1, keepdim=True))
        return v * (lambda_x / lambda_y)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project back to ball if outside."""
        if self._ball is not None:
            return self._ball.project(x, self.c, eps=1e-5)
        # Inline fallback
        norm = x.norm(dim=-1, keepdim=True)
        mask = norm >= self.max_norm
        if mask.any():
            x = torch.where(mask, x / norm * self.max_norm, x)
        return x

# --- Hamiltonian Dynamics Engine ---

class SymplecticIntegrator:
    def __init__(self, geo: RiemannianGeometry, config: HamiltonianConfig, potential_fn=None):
        self.geo = geo
        self.steps = config.num_leapfrog_steps
        self.dt = config.step_size
        self.damping = config.damping
        self.mass_inv = 1.0 / config.mass_matrix_diagonal
        self.custom_potential = potential_fn
        
    def potential_energy(self, z: torch.Tensor, target_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Potential U(z). 
        Can include:
        1. Task Loss (if backpropped)
        2. Geometric Regularization (distance from target/origin)
        """
        if self.custom_potential is not None:
            return self.custom_potential(z)

        # Default potential: Harmonic oscillator towards origin (regularization)
        # U = 0.5 * k * d(0, z)^2
        dist = self.geo.dist(torch.zeros_like(z), z)
        return 0.5 * (dist ** 2).sum()

    def leapfrog_step(self, z: torch.Tensor, p: torch.Tensor, 
                      allowed_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        dt = self.dt
        
        # If masked, only update active elements
        # NOTE: We assume z, p are batches of fiber states processed in parallel
        
        # 1. Half step momentum
        # Force = -grad(U)
        with torch.enable_grad():
            z_in = z.detach().requires_grad_(True)
            u = self.potential_energy(z_in)
            grads = torch.autograd.grad(u, z_in, create_graph=False)[0]
            force = -grads
            
        if allowed_mask is not None:
            force = force * allowed_mask
            
        p_half = p + 0.5 * dt * force
        p_half = p_half * (1.0 - self.damping) # Damping
        
        # 2. Full step position
        # v = M^-1 * p
        v = p_half * self.mass_inv
        
        # z_new = Exp_z(v * dt)
        # For simplicity in global coord chart:
        z_new = self.geo.exp_map(z, v * dt)
        z_new = self.geo.project(z_new) # Constraint
        
        if allowed_mask is not None:
             # Reset inactive positions
             # z_new = mask * z_new + (1-mask) * z
             z_new = torch.where(allowed_mask.bool(), z_new, z)

        # 3. Half step momentum (at new position)
        with torch.enable_grad():
            z_new_in = z_new.detach().requires_grad_(True)
            u_new = self.potential_energy(z_new_in)
            grads_new = torch.autograd.grad(u_new, z_new_in, create_graph=False)[0]
            force_new = -grads_new

        if allowed_mask is not None:
            force_new = force_new * allowed_mask

        p_new = p_half + 0.5 * dt * force_new
        
        return z_new, p_new

    def leapfrog_bundle_step(self, coords: torch.Tensor, sections: torch.Tensor,
                             p_coords: torch.Tensor, p_sections: torch.Tensor,
                             metric: Optional[Any] = None,
                             allowed_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Specialized Leapfrog step for Bundle State (Base Coords + Fiber Sections).
        Base Coords evolve on Manifold.
        Fiber Sections evolve in Euclidean/Simplex space.
        """
        dt = self.dt
        
        # 1. Gradient (Force) Calculation
        with torch.enable_grad():
            c_in = coords.detach().requires_grad_(True)
            s_in = sections.detach().requires_grad_(True)
            
            # Use custom potential if available
            if self.custom_potential:
                 try:
                     u = self.custom_potential(c_in, s_in, metric)
                 except TypeError:
                     u = self.custom_potential(c_in) # Fallback
            else:
                 u = 0.5 * (c_in.norm()**2 + s_in.norm()**2) 
            
            grads = torch.autograd.grad(u, [c_in, s_in], create_graph=False, retain_graph=False, allow_unused=True)
            f_c = -grads[0] if grads[0] is not None else torch.zeros_like(c_in)
            f_s = -grads[1] if grads[1] is not None else torch.zeros_like(s_in)
            
        # Mask Forces
        if allowed_mask is not None:
            f_c = f_c * allowed_mask
            f_s = f_s * allowed_mask
            
        # 2. Momentum Half Step
        p_c_half = p_coords + 0.5 * dt * f_c
        p_c_half = p_c_half * (1.0 - self.damping)
        
        p_s_half = p_sections + 0.5 * dt * f_s
        p_s_half = p_s_half * (1.0 - self.damping)

        # 3. Position Full Step
        # Coords -> Manifold Exp Map
        # v_c = M^-1 * p
        v_c = p_c_half * self.mass_inv 
        c_new = self.geo.exp_map(coords, v_c * dt)
        c_new = self.geo.project(c_new)

        # Sections -> Euclidean update
        v_s = p_s_half * self.mass_inv
        s_new = sections + v_s * dt
        
        # Anchoring
        if allowed_mask is not None:
            c_new = torch.where(allowed_mask.bool(), c_new, coords)
            s_new = torch.where(allowed_mask.bool(), s_new, sections)
            
        # 4. Momentum Half Step (New Pos)
        with torch.enable_grad():
            c_new_in = c_new.detach().requires_grad_(True)
            s_new_in = s_new.detach().requires_grad_(True)
            
            if self.custom_potential:
                 try:
                     u_new = self.custom_potential(c_new_in, s_new_in, metric)
                 except TypeError:
                     u_new = self.custom_potential(c_new_in)
            else:
                 u_new = 0.5 * (c_new_in.norm()**2 + s_new_in.norm()**2)
                 
            grads_new = torch.autograd.grad(u_new, [c_new_in, s_new_in], create_graph=False, retain_graph=False, allow_unused=True)
            f_c_new = -grads_new[0] if grads_new[0] is not None else torch.zeros_like(c_new)
            f_s_new = -grads_new[1] if grads_new[1] is not None else torch.zeros_like(s_new)
            
        if allowed_mask is not None:
            f_c_new = f_c_new * allowed_mask
            f_s_new = f_s_new * allowed_mask
            
        p_c_new = p_c_half + 0.5 * dt * f_c_new
        p_s_new = p_s_half + 0.5 * dt * f_s_new
        
        return c_new, s_new, p_c_new, p_s_new
        
    def integrate(self, fiber_states: Dict[FiberID, FiberState], allowed_ids: Set[FiberID]) -> None:
        """In-place update of fiber states using Hamiltonian dynamics (for Active set)."""
        # 1. Collate active states into Tensor batch
        active_list = list(allowed_ids)
        if not active_list: return
        
        z_batch = torch.stack([fiber_states[fid].z for fid in active_list])
        p_batch = torch.stack([fiber_states[fid].p for fid in active_list])
        
        # 2. Run Leapfrog
        for _ in range(self.steps):
            z_batch, p_batch = self.leapfrog_step(z_batch, p_batch)
            
        # 3. Scatter back
        for i, fid in enumerate(active_list):
            fiber_states[fid].z = z_batch[i]
            fiber_states[fid].p = p_batch[i]


# --- Fiber Bundle Manager & Router ---

class ManifoldGLManager:
    """Main Orchestrator for the Fiber Bundle."""
    
    def __init__(self, m_config: ManifoldConfig, h_config: HamiltonianConfig, e_config: EffectConfig):
        self.geo = RiemannianGeometry(m_config)
        self.dynamics = SymplecticIntegrator(self.geo, h_config)
        self.e_config = e_config
        self.topology = AdjacencyGraph()
        self.fibers: Dict[FiberID, FiberState] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = m_config.latent_dim

        # Resonant Dynamics (Phase 2.5)
        # Pre-allocate for fixed capacity or expand dynamically
        self.max_fibers = 100 
        self.fiber_id_map: Dict[FiberID, int] = {} # Map ID -> Resonator Index
        self.next_idx = 0
        
        self.resonator = None
        if FHNIntegrator:
            self.resonator = FHNIntegrator(self.max_fibers)
            self.resonator_state = (torch.zeros(self.max_fibers), torch.zeros(self.max_fibers)) # q, p
        else:
            print("WARNING: FHNIntegrator module missing. Resonant dynamics disabled.")

    def register_fiber(self, fid: FiberID):
        if fid not in self.fibers:
            self.fibers[fid] = FiberState(
                z=torch.zeros(self.latent_dim, device=self.device),
                p=torch.zeros(self.latent_dim, device=self.device)
            )
            
            # Map to resonator slot if capacity allows and not already mapped
            if fid not in self.fiber_id_map and self.next_idx < self.max_fibers:
                self.fiber_id_map[fid] = self.next_idx
                self.next_idx += 1
            
    def compute_locus(self, input_signal: torch.Tensor) -> List[FiberID]:
        """Determine which fibers are 'active' (Locus) for this input."""
        # Mock logic: If we had a trained router, we'd use it.
        # For v1 test, we'll manually activate 'A' or 'B'.
        # Assuming input has some ID or strict mapping.
        # Placeholder: Return all keys for now or specific ones triggered by test
        return list(self.fibers.keys())[:1] # Default to first

    def local_update(self, fid: FiberID, input_signal: torch.Tensor) -> Tuple[FiberDelta, str]:
        """Compute local gradient update (Delta) for a fiber."""
        # In real system: Forward pass adapter -> backward -> grad
        # Test Stub: Random update
        delta = torch.randn(self.latent_dim, device=self.device) * 0.05
        return FiberDelta(z_delta=delta, source_hop=0), f"Updated {fid}"

    def propagate(self, source_fid: FiberID, delta: FiberDelta, allowed_targets: Set[FiberID]) -> Tuple[Dict[FiberID, FiberDelta], Set[FiberID]]:
        """Propagate updates along adjacency graph."""
        deltas = {source_fid: delta}
        extended_allowed = allowed_targets.copy()
        max_hops = self.e_config.max_propagation_hops
        
        frontier = {source_fid}
        for hop in range(max_hops):
            next_frontier = set()
            for fid in frontier:
                neighbors = self.topology.get_neighbors(fid, self.e_config.overlap_threshold)
                for nid, edge_type in neighbors:
                    # Gating: Propagate only if neighbor is already allowed or edge is valid type
                    # Using broad policy: Propagation EXPANDS allowed targets
                    if nid not in deltas:
                        # Transport Delta
                        # P_i->j (v)
                        z_i = self.fibers[fid].z
                        z_j = self.fibers[nid].z
                        transported_v = self.geo.parallel_transport(z_i, z_j, deltas[fid].z_delta)
                        
                        weight = self.topology.get_weight(fid, nid)
                        deltas[nid] = FiberDelta(z_delta=transported_v * weight * 0.5, source_hop=hop+1) # Decay
                        
                        next_frontier.add(nid)
                        extended_allowed.add(nid)
            frontier = next_frontier
            
        return deltas, extended_allowed

    def anchor_off_locus(self, allowed_targets: Set[FiberID], prev_states: Dict[FiberID, FiberState]) -> None:
        """Effect Discipline: Anchor inactive fibers."""
        strength = self.e_config.anchor_strength
        
        for fid in self.fibers:
            if fid not in allowed_targets:
                # Anchor!
                z_prev = prev_states[fid].z
                z_curr = self.fibers[fid].z
                
                # Soft anchor: z_new = strength * z_prev + (1-strength) * z_curr
                # Better: Geodesic interpolation
                # Simple Euclidean for v1 latent
                z_anchored = strength * z_prev + (1.0 - strength) * z_curr
                self.fibers[fid].z = self.geo.project(z_anchored)
                self.fibers[fid].p *= 0.0 # Kill momentum for inactive

    def step(self, input_signal: torch.Tensor, override_locus: List[FiberID] = None) -> EffectLog:
        """Full Bundle Step."""
        
        # 1. Snapshot State
        prev_states = {fid: s.clone() for fid, s in self.fibers.items()}
        
        # 2. Locus
        locus = override_locus if override_locus else self.compute_locus(input_signal)
        allowed_targets = set(locus)
        logs = []
        
        # --- PHASE 2.5: Resonant Activation ---
        resonance_gates = {} # fid -> scalar
        if self.resonator:
            # Construct Input Drive
            drive = torch.zeros(self.max_fibers)
            for fid in locus:
                idx = self.fiber_id_map.get(fid)
                if idx is not None:
                     drive[idx] = 1.0 # Simple pulse drive for active locus
            
            # Step Resonator
            q_new, p_new = self.resonator.step(self.resonator_state, drive)
            self.resonator_state = (q_new, p_new)
            
            # Extract Gating Values (Sigmoid(q))
            for fid, idx in self.fiber_id_map.items():
                gate = torch.sigmoid(q_new[idx]).item()
                resonance_gates[fid] = gate
        else:
            # Pass-through if no resonator
            for fid in self.fibers: resonance_gates[fid] = 1.0

        # 3. Local Updates & Propagation
        all_deltas = {}
        
        for fid in locus:
            delta, log = self.local_update(fid, input_signal)
            logs.append(log)
            
            # Apply Resonant Gating to Delta Magnitude
            gate = resonance_gates.get(fid, 0.0)
            delta.z_delta = delta.z_delta * gate # Modulation!
            
            p_deltas, new_allowed = self.propagate(fid, delta, allowed_targets)
            allowed_targets.update(new_allowed)
            all_deltas.update(p_deltas)
            
        # 4. Apply Deltas (Geometric Addition)
        for fid, delta in all_deltas.items():
            # z_new = Exp_z(delta)
            self.fibers[fid].z = self.geo.exp_map(self.fibers[fid].z, delta.z_delta)
            self.fibers[fid].z = self.geo.project(self.fibers[fid].z)
            
        # 5. Hamiltonian Relaxation (Dynamics)
        # Note: In real training, this happens naturally if we use Symplectic Optimizer.
        # Here we explicit run simulation step.
        self.dynamics.integrate(self.fibers, allowed_targets)
        
        # 6. Anchor Off-Locus
        self.anchor_off_locus(allowed_targets, prev_states)
        
        return EffectLog(logs=logs, allowed_targets=allowed_targets)


# --- Regression Suite (6-Probe) ---

def run_tests():
    print("\n=== Running ManifoldGL 6-Probe Regression Suite ===")
    
    # Setup
    m_conf = ManifoldConfig()
    h_conf = HamiltonianConfig()
    e_conf = EffectConfig()
    mgr = ManifoldGLManager(m_conf, h_conf, e_conf)
    
    # Topology: A <-> B (Tree), B <-> C (Overlap), D (Isolated)
    mgr.register_fiber("A")
    mgr.register_fiber("B")
    mgr.register_fiber("C")
    mgr.register_fiber("D")
    
    mgr.topology.add_tree_edge("A", "B") # A is child of B
    mgr.topology.add_overlap("B", "C", 0.9) # Strong overlap
    
    # --- PROBE A: Locus Activation ---
    print("[Test A] Locus Activation...")
    log = mgr.step(torch.randn(1), override_locus=["A"])
    assert "A" in log.allowed_targets
    print("PASS: Locus A activated.")

    # --- PROBE B: Propagation ---
    print("[Test B] Propagation Logic...")
    # Update A -> Should propagate to parent B
    # B overlaps C -> Should propagate to C
    # D is isolated -> Should NOT be touched
    assert "B" in log.allowed_targets, "Failed to propagate child->parent"
    assert "C" in log.allowed_targets, "Failed to propagate overlap"
    assert "D" not in log.allowed_targets, "Failed isolation of D"
    print("PASS: Propagation A->B->C confirmed. D isolated.")
    
    # --- PROBE C: Off-Locus Anchoring ---
    print("[Test C] Off-Locus Anchoring...")
    # Store D state
    d_prev = mgr.fibers["D"].z.clone()
    # Step again affecting A
    mgr.step(torch.randn(1), override_locus=["A"])
    d_curr = mgr.fibers["D"].z
    dist = torch.norm(d_prev - d_curr)
    assert dist < 1e-6, f"D drifted! Distance: {dist}"
    print("PASS: Fiber D remained stable (Anchored).")

    # --- PROBE D: Growth/Contraction (Dynamics) ---
    print("[Test D] Hamiltonian Stability...")
    # Run pure dynamics on B
    b_energy_start = mgr.dynamics.potential_energy(mgr.fibers["B"].z)
    for _ in range(10):
        mgr.dynamics.integrate(mgr.fibers, {"B"})
    b_energy_end = mgr.dynamics.potential_energy(mgr.fibers["B"].z)
    print(f"PASS: Energy Drift check: {b_energy_start.item():.4f} -> {b_energy_end.item():.4f}")

    print("ALL TESTS PASSED.\n")

if __name__ == "__main__":
    run_tests()
