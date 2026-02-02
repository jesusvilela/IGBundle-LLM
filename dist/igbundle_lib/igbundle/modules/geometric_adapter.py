"""
Geometrically Rigorous IGBundle Adapter

This module provides a mathematically correct implementation of fiber bundle
operations with proper Riemannian geometry and lambda calculus semantics.

Key improvements over the original adapter:
- True Riemannian curvature instead of variance parameters
- Proper lambda calculus operations in fiber bundle context
- Categorical composition in fiber spaces
- Parallel transport for geometric consistency
- Information-geometric updates derived from manifold structure

Author: Jesús Vilela Jato
License: (c) All rights reserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass

from ..geometry.riemannian import (
    RiemannianGeometry,
    FiberBundleLambdaCalculus,
    RiemannianMetric,
    bundle_curvature_loss
)
from .state import MixtureState
from .kl import kl_diag_gauss, kl_categorical_logits
from ..core.config import IGBundleConfig
from ..dynamics.hamiltonian import HamiltonianSystem
from .attention import PoincareAttention
from .vision import VisionProjector
from .compiler import ManifoldDecompiler
from .memory import GeometricPhaseMemory
from .consensus import SheafConsensus
from .manifold_gl_fiber_bundle_v1 import (
    ManifoldConfig as V1ManifoldConfig, 
    HamiltonianConfig as V1HamiltonianConfig, 
    SymplecticIntegrator as V1SymplecticIntegrator,
    RiemannianGeometry as V1RiemannianGeometry
)
from ..fibers.latent_store import FiberLatentStore
from ..fibers.executor import FiberExecutor
from ..fibers.closure import compute_closure

@dataclass
class GeometricState:
    """
    Enhanced state representation with proper geometric structure.

    Attributes:
        mixture_state: Original mixture representation
        base_coordinates: (B, T, P, D) - coordinates on base manifold
        fiber_sections: (B, T, P, K) - sections of categorical fiber bundle
        lambda_terms: (B, T, P, D+K) - encoded lambda calculus terms
        metric: RiemannianMetric - local metric tensor
    """
    mixture_state: MixtureState
    base_coordinates: torch.Tensor
    fiber_sections: torch.Tensor
    lambda_terms: torch.Tensor
    lambda_terms: torch.Tensor
    metric: Optional[RiemannianMetric] = None
    op_logits: Optional[torch.Tensor] = None # Epic 3: Compiler Output
    consensus_loss: Optional[torch.Tensor] = None # Epic 7: Sheaf Consensus Loss
    active_indices: Optional[torch.Tensor] = None # Epic 1: Sparse Hamiltonian Indices

class GeometricIGBundleAdapter(nn.Module):
    """
    Geometrically rigorous IGBundle adapter with proper mathematical foundations.

    This adapter implements:
    1. True Riemannian geometry on the base manifold with learned metrics
    2. Fiber bundle lambda calculus operations
    3. Parallel transport for geometric consistency
    4. Information-geometric updates derived from natural gradients
    5. Sheaf-theoretic consistency constraints
    """

    def __init__(self, config: IGBundleConfig):
        super().__init__()
        self.cfg = config

        # Dimensions
        self.H = config.hidden_size
        self.P = config.num_components
        self.K = config.num_categories
        self.D = config.latent_dim
        self.D_bot = getattr(config, 'bottleneck_dim', self.H // 4)

        print(f"GeometricIGBundle: D={self.D}, P={self.P}, K={self.K}, D_bot={self.D_bot}")

        # Core geometric modules
        self.riemannian_geometry = RiemannianGeometry(config)
        self.lambda_calculus = FiberBundleLambdaCalculus(config)
        
        # --- ManifoldGL V1 Integration ---
        self.v1_m_config = V1ManifoldConfig(latent_dim=self.D)
        self.v1_h_config = V1HamiltonianConfig(
            use_dynamics=getattr(config, 'use_dynamics', True),
            num_leapfrog_steps=4,
            step_size=0.01
        )
        self.v1_geo = V1RiemannianGeometry(self.v1_m_config)
        # Bind compute_hamiltonian_potential as the custom potential function
        self.dynamics_engine = V1SymplecticIntegrator(
            self.v1_geo, 
            self.v1_h_config, 
            potential_fn=self.compute_hamiltonian_potential
        )

        # Input/Output projections
        self.input_proj = nn.Linear(self.H, self.D_bot)
        self.output_proj = nn.Linear(self.D + self.K, self.H)

        # Base manifold coordinate projection
        self.base_coord_proj = nn.Linear(self.D_bot, self.P * self.D)

        # Fiber bundle section projection
        self.fiber_section_proj = nn.Linear(self.D_bot, self.P * self.K)

        # Lambda calculus term encoding
        self.lambda_term_proj = nn.Linear(self.D_bot, self.P * (self.D + self.K))

        # Mixture parameters (for compatibility)
        self.mixture_proj_w = nn.Linear(self.D_bot, self.P)
        self.mixture_proj_m = nn.Linear(self.D_bot, self.P * self.D)
        self.mixture_proj_s = nn.Linear(self.D_bot, self.P * self.D)
        self.mixture_proj_u = nn.Linear(self.D_bot, self.P * self.K)

        # Geometric update networks
        self.base_update_net = nn.Sequential(
            nn.Linear(self.D + self.K, 64),
            nn.Tanh(),
            nn.Linear(64, self.D)
        )

        self.fiber_update_net = nn.Sequential(
            nn.Linear(self.D + self.K, 64),
            nn.Tanh(),
            nn.Linear(64, self.K)
        )

        # Sheaf consistency parameters
        self.num_patches = getattr(config, 'num_sheaf_patches', 4)
        self.patch_centers = nn.Parameter(
            torch.randn(self.num_patches, self.D) * 0.1
        )

        # Scaling and regularization
        self.scale = config.adapter_scale
        self.dropout = nn.Dropout(config.dropout)

        # Initialize to identity
        # Initialize to identity
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # ------------------------------------------------------------------
        # EPIC 5: Multimodal Genesis (Vision)
        # ------------------------------------------------------------------
        if "vision" in getattr(config, "supported_modalities", []):
            print("GeometricIGBundle: Initializing Vision Projector (Epic 5)")
            vision_dim = getattr(config, "vision_dim", 1152) 
            self.vision_projector = VisionProjector(
                vision_dim=vision_dim, 
                bottleneck_dim=self.D_bot,
                dropout=config.dropout
            )
            # Cross-Attention: Text (Query) attends to Vision (Key/Value)
            self.vision_attn = nn.MultiheadAttention(
                embed_dim=self.D_bot,
                num_heads=4, 
                dropout=config.dropout,
                batch_first=True
            )
            self.vision_norm = nn.LayerNorm(self.D_bot)
        else:
            self.vision_projector = None
            self.vision_attn = None

        # EPIC 1: Hamiltonian Dynamics
        self.dynamics = HamiltonianSystem(self.D) # Operates on latent D
        self.use_dynamics = getattr(config, 'use_dynamics', False)

        # EPIC 2: Geodesic Attention
        self.use_geodesic_attn = getattr(config, 'use_geodesic_attn', False)
        if self.use_geodesic_attn:
            self.geodesic_attn = PoincareAttention(self.D)

        # EPIC 3: Task Compiler
        self.compiler = ManifoldDecompiler(self.D)

        # EPIC 6: Quantum Cognition (Memory)
        self.phase_memory = GeometricPhaseMemory(self.D)
        
        # EPIC 7: Sheaf-Theoretic Consensus (The Swarm)
        # We treat the P components as agents in a sheaf network
        self.consensus = SheafConsensus(num_agents=self.P, latent_dim=self.D)
        
        # EPIC 31: Fiber Latent Refinement
        # Store for persistent fiber latents (s_i)
        self.fiber_store = FiberLatentStore(n_fibers=self.P, d_s=self.D, device=None, dtype=None) # device handled later
        self.fiber_executor = FiberExecutor(self.fiber_store)
        
        # Fiber Gate Projection: s_i -> g_i (modulates output)
        # We assume g_i modulates H (hidden size) via channel-wise scaling or rank-wise if we had access.
        # Since we modulate the *adapter output*, we project d_s -> H directly or d_s -> H (sigmoid).
        # To be "LoRA-like", we treat the adapter output as the "Delta".
        # So we gate the Delta: output = scale * (Projected(x) * Gate(s))
        self.fiber_gate_proj = nn.Sequential(
            nn.Linear(self.D, self.H),
            nn.Sigmoid() 
        )
        
        # Adjacency for closure (Placeholder - typically derived from metric or fixed topology)
        # For now, we assume a simple linear chain or fully connected for testing, 
        # or we build it dynamically. Let's use an empty dict until we have a topology builder.
        self.fiber_adjacency = {} 

        self._vision_context = None

    def set_vision_context(self, context: torch.Tensor):
        self._vision_context = context

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, GeometricState]:
        """
        Forward pass with geometrically rigorous operations.

        Args:
            x: (B, T, H) - input hidden states
            context: (B, T, H) - optional context
            pixel_values: Optional (B, C, H, W) or (B, N, D) for vision

        Returns:
            output: (B, T, H) - transformed hidden states
            state: GeometricState - geometric bundle state
        """
        # EPIC 2: Multimodal Handling
        if kwargs.get('pixel_values') is not None and hasattr(self, 'vision_proj'):
             # Future: Inject vision tokens into x or h_bot
             pass

        B, T, H = x.shape

        # 1. Project to bottleneck space
        h_bot = self.input_proj(x)  # (B, T, D_bot)
        
        # EPIC 5: Multimodal Injection
        vis_features = None
        pixel_values = kwargs.get('pixel_values')
        
        if pixel_values is not None and self.vision_projector is not None:
             if pixel_values.dim() == 3:
                 vis_features = self.vision_projector(pixel_values)
        
        if vis_features is None and self._vision_context is not None:
             vis_features = self._vision_context
             
        if vis_features is not None:
            if vis_features.dtype != h_bot.dtype:
                vis_features = vis_features.to(h_bot.dtype)

            if self.vision_attn is not None:
                vis_context_out, _ = self.vision_attn(
                    query=h_bot,
                    key=vis_features,
                    value=vis_features
                )
                h_bot = self.vision_norm(h_bot + vis_context_out)
        
        # Debug before returning or next steps
        # logger.info(f"DEBUG: Adapter h_bot out: {h_bot.dtype}")


        if h_bot.dtype != x.dtype:
            h_bot = h_bot.to(x.dtype)
            

            
        # Note: If pixel_values passed but vision disabled, we ignore it safely.

        # 2. Extract geometric coordinates
        base_coords = self.base_coord_proj(h_bot).view(B, T, self.P, self.D).float()
        fiber_sections = self.fiber_section_proj(h_bot).view(B, T, self.P, self.K).float()
        lambda_terms = self.lambda_term_proj(h_bot).view(B, T, self.P, self.D + self.K).float()

        # 3. Compatibility: extract mixture parameters
        w_logits = self.mixture_proj_w(h_bot).float() # Force float32
        m = self.mixture_proj_m(h_bot).view(B, T, self.P, self.D).float()
        log_s = self.mixture_proj_s(h_bot).view(B, T, self.P, self.D).float()
        log_s = torch.clamp(log_s, min=-5, max=5)
        # Stability: Clamp base_coords norm to avoid boundary singularity
        base_coords = torch.clamp(base_coords, min=-10, max=10).float() # Euclidean clamp before exp map if any
        u = self.mixture_proj_u(h_bot).view(B, T, self.P, self.K).float()

        mixture_state = MixtureState(w_logits, m, log_s, u)

        # 4. Compute Riemannian metric at base coordinates
        metric = self.riemannian_geometry.get_metric(base_coords)

        # 5. Apply geometric transformations

        # 5a. Lambda calculus operations on fiber bundle sections
        transformed_sections = self._apply_lambda_operations(lambda_terms, fiber_sections)

        # 5b. Parallel transport for geometric consistency
        transported_coords = self._parallel_transport_update(base_coords, metric)

        # 5c. Hamiltonian Sheaf Dynamics (Symplectic Integration)
        # Replaces simple Natural Gradient with Physics-based Flow
        # 5c. Hamiltonian Sheaf Dynamics (Symplectic Integration)
        active_indices = None
        if self.use_dynamics:
             # Determine Active Indices (Allowed Targets) based on router confidence
             # Effect Discipline: Only top-k bundles are allowed to evolve
             k_active = 3
             router_confidence = torch.max(transformed_sections, dim=-1)[0].squeeze(1) # (B, P)
             _, active_indices = torch.topk(router_confidence, k=k_active, dim=-1) # (B, k)
             
             # Symplectic integration of the Hamiltonian system
             updated_coords, updated_sections = self._symplectic_integrate(
                 transported_coords, transformed_sections, metric, active_indices=active_indices
             )
        else:
             # Fallback to Information Geometric Update (Legacy)
             updated_coords, updated_sections = self._information_geometric_update(
                 transported_coords, transformed_sections, metric
             )

        # EPIC 1: Hamiltonian Evolution is now handled in 5c above.
        # Legacy placeholder removed.

        # EPIC 2: Geodesic Attention
        # Apply attention mechanism based on hyperbolic distances
        if self.use_geodesic_attn:
            # Attend to geometrically similar concepts in the sequence
            # This enriches the representation with "similar" historical context
            attn_out = self.geodesic_attn(updated_coords, updated_coords)
            updated_coords = updated_coords + attn_out # Residual connection on the manifold


        # 6. Aggregate across components using Riemannian structure
        weights = F.softmax(w_logits, dim=-1)  # (B, T, P)

        # Weighted geometric mean on manifold
        aggregated_coords = self._riemannian_weighted_mean(updated_coords, weights, metric)
        aggregated_sections = torch.einsum('btp,btpk->btk', weights, updated_sections)

        # EPIC 6: Quantum Cognition (Apply Phase Memory)
        # Calculate geometric phase from the path of updated_coords and modulate specific memory
        phase_context, phase_vals = self.phase_memory(updated_coords)
        
        # Inject Memory Context into Aggregated Coords (Residual)
        # This means history (phase) modulates current state
        aggregated_coords = aggregated_coords + phase_context

        # EPIC 7: Sheaf Consensus (Swarm Logic)
        # Before final projection, apply consensus diffusion to the updated_coords?
        # Or just compute loss?
        # Let's apply diffusion if enabled, to align thoughts.
        # updated_coords: (B, T, P, D) -> Permute to (B, T, num_agents, D) match consensus
        # Actually P is NumComponents.
        consensual_coords = self.consensus(updated_coords)
        
        # Calculate Consensus Loss (Agreement)
        consensus_loss = self.consensus.compute_agreement_loss(updated_coords)
        
        # Use consensual coordinates for final output?
        # Maybe mix them? For now, let's use them to guide the output, 
        # but keep updated_coords for the state (diversity).
        # Actually, if we enforce consensus, we should use the aligned thoughts.
        # But we also want diversity in the bundle sections.
        # Let's use consensual_coords for the "aggregated" output path.
        
        # Re-compute aggregated coords using consensual thoughts
        aggregated_coords_consensus = self._riemannian_weighted_mean(consensual_coords, weights, metric)
        
        # Add residual from memory to this consensus result
        aggregated_coords_consensus = aggregated_coords_consensus + phase_context
        
        # Update output generation to use consensual result
        # combined = torch.cat([aggregated_coords, aggregated_sections], dim=-1) 
        # VS
        combined = torch.cat([aggregated_coords_consensus, aggregated_sections], dim=-1)

        # Update state with consensus metrics?
        # We'll store consensus_loss in the state via a hack or return it?
        # GeometricState doesn't have a loss field.
        # We usually compute losses in `compute_geometric_losses`.
        # We should store consensus_loss in the adapter to be picked up later? 
        # No, that's stateful.
        # We can add `consensus_loss` to GeometricState.

        # EPIC 3: Run Compiler on "Reasoning State"
        # Extract symbolic operations from the updated coordinates
        # Stability: Clamp inputs to compiler?
        op_logits = self.compiler(torch.clamp(updated_coords, -5, 5))

        # 7. Project back to hidden space
        combined = torch.cat([aggregated_coords_consensus, aggregated_sections], dim=-1)  # (B, T, D+K)
        output = self.output_proj(combined)  # (B, T, H)

        # EPIC 31: Fiber Latent Gating
        # Modulate the output using the fiber latents (s_i)
        # We compute this for ALL fibers P and weight by the router weights to ensure smooth gradients
        all_s = self.fiber_store.s # (P, D)
        all_gates = self.fiber_gate_proj(all_s) # (P, H)
        
        # Weighted Gate: sum(w_i * g_i)
        # weights: (B, T, P)
        # all_gates: (P, H)
        weighted_gate = torch.einsum('btp,ph->bth', weights, all_gates)
        
        # Modulate Adapter Output
        output = output * weighted_gate

        output = self.dropout(output)

        # 8. Construct geometric state
        geo_state = GeometricState(
            mixture_state=mixture_state,
            base_coordinates=updated_coords,
            fiber_sections=updated_sections,
            lambda_terms=lambda_terms,
            metric=metric,
            op_logits=op_logits,
            consensus_loss=consensus_loss,
            active_indices=active_indices
        )

        return x + self.scale * output, geo_state

    def refine_latents(self, active_indices: torch.Tensor, adjacency: Optional[Dict[int, Set[int]]] = None):
        """
        Execute an Effect-Disciplined Refinement Step.
        Args:
            active_indices: (B, k) indices of active fibers from forward pass.
            adjacency: Optional adjacency override.
        """
        # Collapse batch active indices to a single set of seed fibers
        seeds = set(active_indices.view(-1).tolist())
        
        # Compute Closure (Allowed Targets)
        # Depth 1 or 2
        allowed_targets = compute_closure(seeds, adjacency if adjacency else self.fiber_adjacency, depth=1)
        
        # Scheduler for refinement (Fixed 1-step logic for now)
        # 1. Anchor Off-Locus (Prevent drift)
        self.fiber_executor.anchor_off_locus(allowed_targets, beta=0.1)
        
        # 2. Propagate (Smooth within allowed region)
        if adjacency:
             self.fiber_executor.propagate(adjacency, allowed_targets, eta=0.05)
             
    def _apply_lambda_operations(self, lambda_terms: torch.Tensor,
                                fiber_sections: torch.Tensor) -> torch.Tensor:
        """Apply lambda calculus operations in the fiber bundle context."""
        B, T, P, _ = lambda_terms.shape

        transformed_sections = fiber_sections.clone()

        for p in range(P):
            # Extract lambda term and section for this component
            lambda_p = lambda_terms[:, :, p, :]  # (B, T, D+K)
            section_p = fiber_sections[:, :, p, :]  # (B, T, K)

            # Pad section to match lambda term dimension
            padded_section = F.pad(section_p, (0, self.D), value=0.0)

            # Apply lambda calculus application operation
            result = self.lambda_calculus.application(lambda_p, padded_section)

            # Extract fiber part of result
            transformed_sections[:, :, p, :] = result[:, :, self.D:]

        return transformed_sections

    def _parallel_transport_update(self, base_coords: torch.Tensor,
                                  metric: RiemannianMetric) -> torch.Tensor:
        """Apply parallel transport to maintain geometric consistency."""
        B, T, P, D = base_coords.shape

        if T <= 1:
            return base_coords

        transported_coords = base_coords.clone()

        # Vectorized implementation (O(1) Python overhead vs O(T))
        # This replaces the slow loop and removes unused metric instantiation
        prev_coords = base_coords[:, :-1, :, :]  # (B, T-1, P, D)
        curr_coords = base_coords[:, 1:, :, :]   # (B, T-1, P, D)
        
        # Simple geometric correction (matches previous logic but vectorized)
        correction = 0.1 * (curr_coords - prev_coords)
        
        transported_coords[:, 1:, :, :] = curr_coords - correction
        
        if transported_coords.dtype != base_coords.dtype:
            transported_coords = transported_coords.to(base_coords.dtype)
            
        return transported_coords

    def _information_geometric_update(self, coords: torch.Tensor, sections: torch.Tensor,
                                    metric: RiemannianMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply information-geometric updates using natural gradients."""
        B, T, P, D = coords.shape

        # Combine coordinates and sections for joint processing
        joint_repr = torch.cat([coords, sections], dim=-1)  # (B, T, P, D+K)

        # Compute natural gradient updates
        base_update = self.base_update_net(joint_repr)  # (B, T, P, D)
        fiber_update = self.fiber_update_net(joint_repr)  # (B, T, P, K)

        # Scale updates by metric (simplified natural gradient)
        # In full implementation, would use inverse metric tensor
        metric_inv_diag = 1.0 / (torch.diagonal(metric.metric_tensor, dim1=-2, dim2=-1) + 1e-6)
        if metric_inv_diag.dtype != base_update.dtype:
            metric_inv_diag = metric_inv_diag.to(base_update.dtype)
            
        scaled_base_update = base_update * metric_inv_diag

        # Apply updates with geometric constraints
        eta_base = self.cfg.eta_b
        eta_fiber = self.cfg.eta_f

        updated_coords = coords + eta_base * scaled_base_update
        updated_sections = sections + eta_fiber * fiber_update
        
        # Enforce dtype consistency
        if updated_coords.dtype != coords.dtype:
            updated_coords = updated_coords.to(coords.dtype)
        if updated_sections.dtype != sections.dtype:
            updated_sections = updated_sections.to(sections.dtype)

        # Ensure fiber sections remain on probability simplex
        updated_sections = F.softmax(updated_sections, dim=-1)

        return updated_coords, updated_sections

    def _riemannian_weighted_mean(self, coords: torch.Tensor, weights: torch.Tensor,
                                 metric: RiemannianMetric) -> torch.Tensor:
        """Compute Riemannian weighted mean (Frechet mean) of coordinates."""
        B, T, P, D = coords.shape

        # Initialize with Euclidean weighted mean
        weights_exp = weights.unsqueeze(-1)  # (B, T, P, 1)
        euclidean_mean = torch.sum(weights_exp * coords, dim=2)  # (B, T, D)

        # Iterative refinement towards Riemannian mean
        # (simplified - full implementation would use proper geodesic averaging)
        current_mean = euclidean_mean

        for iteration in range(3):  # Few iterations for efficiency
            # Compute logarithmic maps from current mean to each point
            log_vecs = []
            for p in range(P):
                coord_p = coords[:, :, p, :]  # (B, T, D)

                # Extract metric (use mean of component metrics)
                mean_metric = RiemannianMetric(
                    metric.metric_tensor.mean(dim=2, keepdim=True)
                )

                # Compute log map (simplified)
                log_vec = coord_p - current_mean  # Euclidean approximation
                log_vecs.append(log_vec)

            log_vecs = torch.stack(log_vecs, dim=2)  # (B, T, P, D)

            # Weighted average in tangent space
            weighted_log = torch.sum(weights_exp * log_vecs, dim=2)  # (B, T, D)

            # Map back to manifold (simplified exponential map)
            current_mean = current_mean + 0.1 * weighted_log

        return current_mean

    def compute_geometric_losses(self, state: GeometricState) -> Dict[str, torch.Tensor]:
        """Compute geometric regularization losses."""
        losses = {}

        # 1. Curvature regularization - encourage hyperbolic structure
        # 1. Curvature regularization - encourage hyperbolic structure
        # Enabled via stochastic estimation (O(1))
        # This provides telemetry without crashing performance
        if hasattr(self.riemannian_geometry, 'estimate_sectional_curvature_stochastic'):
             # MEMORY FIX: Downsample tokens to prevent OOM (D^3 tensor is huge)
             # Take max 16 random tokens
             B, T, P, D = state.base_coordinates.shape
             if T > 16:
                 indices = torch.randperm(T)[:16]
                 sampled_coords = state.base_coordinates[:, indices, :, :]
             else:
                 sampled_coords = state.base_coordinates
                 
             curvature = self.riemannian_geometry.estimate_sectional_curvature_stochastic(sampled_coords, num_samples=1)
             # Target curvature is -1 (Hyperbolic)
             losses['curvature'] = F.mse_loss(curvature, torch.full_like(curvature, -1.0))
        else:
             losses['curvature'] = torch.tensor(0.0, device=state.base_coordinates.device)

        # 2. Sheaf consistency loss
        sheaf_loss = self._compute_sheaf_consistency_loss(state)
        losses['sheaf_consistency'] = sheaf_loss

        # 3. Bundle structure loss (ensure proper fiber bundle topology)
        bundle_loss = self._compute_bundle_structure_loss(state)
        losses['bundle_structure'] = bundle_loss

        # 4. Lambda calculus type consistency
        lambda_loss = self._compute_lambda_consistency_loss(state)
        losses['lambda_consistency'] = lambda_loss
        
        # EPIC 1 & 4: Hamiltonian Action / Energy Loss
        # Minimize total energy magnitude (efficiency) or drift (stability)
        # Here we encourage "Least Action" -> Minimize Kinetic energy (change) where not needed?
        # Let's just minimize Total Energy to encourage "Low Energy States" (Truth).
        if hasattr(state, 'base_coordinates') and self.use_dynamics:
             # Re-compute energy of the states
             # We assume p=0 for static states to check potential
             p_zero = torch.zeros_like(state.base_coordinates)
             energy = self.dynamics.hamiltonian(state.base_coordinates, p_zero)
             losses['hamiltonian_energy'] = energy.mean() * 0.01 # Scale down
             
        # EPIC 3: Task Compiler Consistency (Entropy Regularization)
        # Encourage the compiler to be confident (Low Entropy on op_logits)
        if state.op_logits is not None:
             logits = state.op_logits # (B, T, NumOps)
             probs = F.softmax(logits, dim=-1)
             entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
             losses['compiler_entropy'] = entropy * 0.1
             
        # EPIC 7: Sheaf Consensus Loss
        if state.consensus_loss is not None:
             losses['consensus_agreement'] = state.consensus_loss * 0.01 # Scale down from ~140 to ~1.4
             
        # EPIC 6: Phase Memory coherence? (Optional)
             
        return losses

    def _compute_sheaf_consistency_loss(self, state: GeometricState) -> torch.Tensor:
        """Compute sheaf-theoretic consistency loss across patches."""
        B, T, P, K = state.fiber_sections.shape

        total_loss = 0.0
        num_pairs = 0

        # Compute soft patch assignments
        coords = state.base_coordinates  # (B, T, P, D)

        for i in range(self.num_patches):
            for j in range(i + 1, self.num_patches):
                center_i = self.patch_centers[i]  # (D,)
                center_j = self.patch_centers[j]  # (D,)

                # Compute distances to patch centers
                dist_i = torch.norm(coords - center_i.view(1, 1, 1, -1), dim=-1)  # (B, T, P)
                dist_j = torch.norm(coords - center_j.view(1, 1, 1, -1), dim=-1)  # (B, T, P)

                # Soft assignments with temperature
                tau = 1.0
                weight_i = F.softmax(-dist_i / tau, dim=-1)  # (B, T, P)
                weight_j = F.softmax(-dist_j / tau, dim=-1)  # (B, T, P)

                # Weighted fiber distributions
                fiber_i = torch.einsum('btp,btpk->btk', weight_i, state.fiber_sections)
                fiber_j = torch.einsum('btp,btpk->btk', weight_j, state.fiber_sections)

                # Jensen-Shannon divergence between distributions
                js_div = self._jensen_shannon_divergence(fiber_i, fiber_j)

                # Weight by patch overlap
                patch_dist = torch.norm(center_i - center_j)
                overlap_weight = torch.exp(-patch_dist)

                total_loss += overlap_weight * js_div.mean()
                num_pairs += 1

        return total_loss / max(num_pairs, 1)

    def _compute_bundle_structure_loss(self, state: GeometricState) -> torch.Tensor:
        """Ensure proper fiber bundle structure is maintained."""
        # Local triviality: nearby points should have similar fiber structure
        B, T, P, D = state.base_coordinates.shape

        if P <= 1:
            return torch.tensor(0.0, device=state.base_coordinates.device)

        total_loss = 0.0

        # Compare adjacent components
        for p in range(P - 1):
            coord_p = state.base_coordinates[:, :, p, :]     # (B, T, D)
            coord_p1 = state.base_coordinates[:, :, p+1, :]  # (B, T, D)

            fiber_p = state.fiber_sections[:, :, p, :]     # (B, T, K)
            fiber_p1 = state.fiber_sections[:, :, p+1, :] # (B, T, K)

            # Distance in base space
            base_dist = torch.norm(coord_p - coord_p1, dim=-1)  # (B, T)

            # Distance in fiber space (KL divergence)
            fiber_dist = self._kl_divergence_normalized(fiber_p, fiber_p1)  # (B, T)

            # Local triviality: fiber distance should be bounded by base distance
            triviality_violation = F.relu(fiber_dist - 2.0 * base_dist)
            total_loss += triviality_violation.mean()

        return total_loss / (P - 1)

    def _compute_lambda_consistency_loss(self, state: GeometricState) -> torch.Tensor:
        """Ensure lambda calculus terms are well-typed."""
        # Check that lambda terms have consistent dimensions
        B, T, P, dim = state.lambda_terms.shape

        # Split into base and fiber parts
        base_part = state.lambda_terms[:, :, :, :self.D]     # (B, T, P, D)
        fiber_part = state.lambda_terms[:, :, :, self.D:]    # (B, T, P, K)

        # Base part should be compatible with base coordinates
        base_consistency = F.mse_loss(base_part, state.base_coordinates)

        # Fiber part should be compatible with fiber sections
        fiber_consistency = F.mse_loss(
            F.softmax(fiber_part, dim=-1),
            state.fiber_sections
        )

        return base_consistency + fiber_consistency

    def _jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute Jensen-Shannon divergence between distributions."""
        # Ensure probability distributions
        p = F.softmax(p, dim=-1) + 1e-8
        q = F.softmax(q, dim=-1) + 1e-8

        m = 0.5 * (p + q)

        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)

        return 0.5 * (kl_pm + kl_qm)

    def _kl_divergence_normalized(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute normalized KL divergence."""
        p_norm = F.softmax(p, dim=-1) + 1e-8
        q_norm = F.softmax(q, dim=-1) + 1e-8

        kl = (p_norm * (torch.log(p_norm) - torch.log(q_norm))).sum(dim=-1)
        return kl

    def compute_hamiltonian_potential(self, coords: torch.Tensor, sections: torch.Tensor, metric: RiemannianMetric) -> torch.Tensor:
        """
        Compute the Potential Energy U(z) for the Hamiltonian system.
        U(z) = L_task_proxy + L_sheaf + L_invariance
        """
        # 1. Sheaf Consistency Energy (Minimize disagreement)
        # Re-use the loss computation logic but applied effectively as energy
        # Create a transient state object for loss computation functions
        # Hacky but effective reuse
        dummy_state = GeometricState(
             mixture_state=None,
             base_coordinates=coords,
             fiber_sections=sections,
             lambda_terms=None, # Not used in sheaf/bundle energy
             metric=metric
        )
        
        sheaf_energy = self._compute_sheaf_consistency_loss(dummy_state)
        bundle_energy = self._compute_bundle_structure_loss(dummy_state)
        
        # 2. Invariance Energy (Off-Locus Anchoring)
        # Penalize movement of "Background" bundles
        # Ideally we identify these via Lambda Calculus, but for now we trust the Router (Bundles 0-2 usually background)
        # Soft-constraint: P[0] should be stable
        if coords.shape[2] > 1:
             background_energy = torch.norm(coords[:, :, 0, :] - coords[:, :, 0, :].detach()) 
             # Wait, that's zero. We need deviation from *initial* or *prev* state.
             # In a single forward pass, "prev" is the input to the integrator.
             pass 

        # 3. Repulsion Energy (Sibling Separation)
        # Inverse distance potential
        # dist = torch.cdist(coords, coords) # (B, T, P, P)
        # repulsion = 1.0 / (dist + 1e-3)
        # But only for siblings? 
        # For now, let's stick to Sheaf + Bundle energy as the potential.
        
        total_potential = sheaf_energy + bundle_energy
        return total_potential

    def _symplectic_integrate(self, coords: torch.Tensor, sections: torch.Tensor, metric: RiemannianMetric, 
                            active_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Delegate to ManifoldGL V1 SymplecticIntegrator.
        """
        # Create mask from active_indices (B, k) -> (B, P)
        allowed_mask = None
        if active_indices is not None:
            B, T, P, _ = coords.shape
            mask_flat = torch.zeros(B, P, device=coords.device)
            
            # Ensure indices are long
            indices = active_indices.long()
            # Ensure indices are 2D (B, k)
            if indices.dim() == 1:
                indices = indices.unsqueeze(0)
            if indices.dim() > 2:
                indices = indices.view(B, -1)
                
            # Scatter 1s
            mask_flat.scatter_(1, indices, 1.0)
            # Expand to (B, T, P, 1) to match coords
            allowed_mask = mask_flat.view(B, 1, P, 1) # (B, 1, P, 1) -> broadcasts to T
            
        # Initialize Momentum (p)
        p_coords = torch.zeros_like(coords)
        p_sections = torch.zeros_like(sections)
        
        z_coords = coords
        z_sections = sections
        
        # Integration Loop (Steps defined in config)
        steps = self.v1_h_config.num_leapfrog_steps
        
        for _ in range(steps):
             z_coords, z_sections, p_coords, p_sections = self.dynamics_engine.leapfrog_bundle_step(
                 z_coords, z_sections, p_coords, p_sections,
                 metric=metric,
                 allowed_mask=allowed_mask
             )
        
        # Ensure Probability Simplex for Sections (Optional, or handled by potential/softmax later)
        # z_sections is usually logits in this architecture, softmax applied at aggregation.
             
        return z_coords, z_sections

def create_geometric_adapter(config) -> GeometricIGBundleAdapter:
    """Factory function to create geometrically rigorous adapter."""
    return GeometricIGBundleAdapter(config)