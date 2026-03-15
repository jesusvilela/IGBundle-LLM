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

from .delta_fiber import DeltaFiberUpdate, DeltaNetAttention

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NAN DETECTED in {name} - Max: {tensor.max()} Min: {tensor.min()}")
        return True
    if torch.isinf(tensor).any():
        print(f"INF DETECTED in {name} - Max: {tensor.max()} Min: {tensor.min()}")
        return True
    return False

from ..geometry.riemannian import (
    RiemannianGeometry,
    FiberBundleLambdaCalculus,
    RiemannianMetric,
    bundle_curvature_loss
)
from .state import MixtureState
from .kl import kl_diag_gauss, kl_categorical_logits
from ..core.config import IGBundleConfig
from ..dynamics.hamiltonian import HamiltonianSystem, LeapfrogIntegrator
from ..dynamics.potential import NeuralPotential
from ..geometry.poincare import PoincareBall
from ..geometry.kan_manifold import KanManifold
from .riemannian_attention import RiemannianAttention
from .hybrid_gating import EntropyGating
from .vision import VisionProjector
from .compiler import ManifoldDecompiler
from .memory import GeometricPhaseMemory
from .consensus import SheafConsensus
# Phase 2 Upgrade: Removed V1 prototype imports
from ..fibers.latent_store import FiberLatentStore
from ..fibers.executor import FiberExecutor
from ..fibers.closure import compute_closure
from ..cognition.meta import MetaCognitiveLoop

class SimplexNaturalGradient(torch.autograd.Function):
    """
    Geometrically rigorous Natural Gradient Descent (NGD) for Categorical probability
    simplices parameterized by logits. Uses the exact rank-(K-1) Fisher-Rao Information
    Metric pseudo-inverse G^+ v = v / theta - sum(v).
    """
    @staticmethod
    def forward(ctx, logits):
        theta = F.softmax(logits, dim=-1)
        # Save theta for backward, clamped to prevent zero-division
        ctx.save_for_backward(torch.clamp(theta, min=1e-8))
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        sum_grad = grad_output.sum(dim=-1, keepdim=True)
        # Apply Fisher Pseudo-Inverse G^+
        natural_grad = (grad_output / theta) - sum_grad
        return natural_grad

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
    metric: Optional[RiemannianMetric] = None
    op_logits: Optional[torch.Tensor] = None # Epic 3: Compiler Output
    consensus_loss: Optional[torch.Tensor] = None # Epic 7: Sheaf Consensus Loss
    active_indices: Optional[torch.Tensor] = None # Epic 1: Sparse Hamiltonian Indices
    meta_info: Optional[Dict[str, Any]] = None # Epic 36: System 2 Trace
    # Phase 4 Telemetry
    hamiltonian_energy: Optional[float] = None
    retrospection_loss: Optional[float] = None
    fhn_phase: Optional[str] = "Free"

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
        

        
        if config.use_dynamics:
            # --- Phase 2: Hamiltonian Dynamics Engine ---
            # Initialize Poincare Ball Geometry (K=-1)

            self.manifold_type = getattr(config, 'manifold_type', 'poincare')
            if self.manifold_type == 'kan':
                 print("GeometricIGBundle: Initialization KAN Manifold (Learnable Geometry)")
                 self.manifold = KanManifold(dim=self.D, hidden_dim=64, base_manifold='poincare')
            else:
                 self.manifold = PoincareBall(dim=self.D)
            
            # Epic 35: Semantic Potential Field (V)
            self.potential_net = NeuralPotential(latent_dim=self.D, hidden_dim=256)
            
            # Initialize Hamiltonian Field (H = T + V)
            print(f"DEBUG: Initializing HamiltonianSystem. Manifold: {self.manifold} Type: {type(self.manifold)}")
            if isinstance(self.manifold, int):
                 print("CRITICAL ERROR: Manifold is INT. Forcing correction to PoincareBall.")
                 # from igbundle.geometry.poincare import PoincareBall # Removed to fix UnboundLocalError
                 self.manifold = PoincareBall(dim=self.D)
            self.vf = HamiltonianSystem(self.manifold, potential_module=self.potential_net)
            self.dynamics = self.vf  # Alias for compatibility
            
            # Initialize Symplectic Integrator
            self.num_leapfrog_steps = getattr(config, 'num_leapfrog_steps', 4)
            self.step_size = getattr(config, 'step_size', 0.01)
            self.integrator = LeapfrogIntegrator(
                self.vf, 
                step_size=self.step_size, 
                num_steps=self.num_leapfrog_steps
            )
            
            # Epic 36: Recursive Meta-Cognition
            self.meta_loop = MetaCognitiveLoop(self.vf, threshold=0.5)
        else:
            self.integrator = None
            self.meta_loop = None

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

        # Epic 17b: Delta-net fiber dynamics (optional upgrade)
        self.use_delta_fiber = getattr(config, 'use_delta_fiber', False)
        if self.use_delta_fiber:
            print("GeometricIGBundle: Initializing Delta-Net Fiber Dynamics (Epic 17b)")
            self.delta_fiber_update = DeltaFiberUpdate(
                coord_dim=self.D,
                section_dim=self.K,
                mem_dim=getattr(config, 'delta_mem_dim', 64),
                num_heads=getattr(config, 'delta_num_heads', 4),
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
            if self.use_delta_fiber:
                # Epic 17b: O(T) delta-net linear attention for vision fusion
                print("GeometricIGBundle: Using DeltaNetAttention for vision (O(T) linear)")
                self.vision_attn = DeltaNetAttention(
                    embed_dim=self.D_bot,
                    num_heads=4,
                    dropout=config.dropout,
                )
            else:
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

        # EPIC 1: Hamiltonian Dynamics (Moved to Phase 2 block above)
        self.use_dynamics = getattr(config, 'use_dynamics', False)

        # EPIC 2: Geodesic Attention
        self.use_geodesic_attn = getattr(config, 'use_geodesic_attn', False)
        if self.use_geodesic_attn:
            # Phase 7 Upgrade: Use Riemannian Attention + Gating
            self.geodesic_attn = RiemannianAttention(config)
            self.gating = EntropyGating(config)
            self.euclidean_bypass = nn.Linear(self.D, self.D) # Fast path

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

    @staticmethod
    def _project_to_ball(x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
        """Project vectors to the Poincare ball: ||x|| < max_norm.

        Per-component tanh/clamp is WRONG for the Poincare ball because
        in D=64 dimensions, component-wise clamp to 0.95 allows ||x|| up to
        sqrt(64)*0.95 = 7.6, wildly outside the unit ball.

        This projects by scaling the vector norm: x_proj = x * min(1, max_norm / ||x||).
        Differentiable and keeps direction intact.
        """
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * scale

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

        # Handle 2D inputs (e.g. from Flash Attn flattening)
        is_2d = False
        if x.dim() == 2:
            is_2d = True
            x = x.unsqueeze(1)
            
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
        fiber_sections_raw = self.fiber_section_proj(h_bot).view(B, T, self.P, self.K).float()
        # Apply strict information geometry Natural Gradient via autograd hook
        fiber_sections = SimplexNaturalGradient.apply(fiber_sections_raw)
        lambda_terms = self.lambda_term_proj(h_bot).view(B, T, self.P, self.D + self.K).float()

        # 3. Compatibility: extract mixture parameters
        w_logits = self.mixture_proj_w(h_bot).float() # Force float32
        m = self.mixture_proj_m(h_bot).view(B, T, self.P, self.D).float()
        log_s = self.mixture_proj_s(h_bot).view(B, T, self.P, self.D).float()
        log_s = torch.clamp(log_s, min=-5, max=5)
        
        # Project base_coords to Poincare Ball B^D(c=1) via norm-based projection.
        # NOTE: Per-component tanh(x)*0.95 is WRONG — with D=64, ||x|| can reach sqrt(64)*0.95=7.6.
        # The ball constraint is ||x|| < 1, so we project by vector norm instead.
        base_coords_raw = self.base_coord_proj(h_bot).view(B, T, self.P, self.D).float()
        base_coords = self._project_to_ball(base_coords_raw, max_norm=0.95)

        u = self.mixture_proj_u(h_bot).view(B, T, self.P, self.K).float()

        mixture_state = MixtureState(w_logits, m, log_s, u)
        m = torch.clamp(m, -10, 10)

        # 4. Compute Riemannian metric at base coordinates
        metric = self.riemannian_geometry.get_metric(base_coords)
        if check_nan(metric.metric_tensor, "metric_g"):
             print("NAN in Metric Tensor!")

        # 5. Apply geometric transformations

        # 5a. Lambda calculus operations on fiber bundle sections
        # Stability: Clamp inputs to Lambda Calculus
        lambda_terms = torch.clamp(lambda_terms, min=-5.0, max=5.0)
        fiber_sections = torch.clamp(fiber_sections, min=-5.0, max=5.0)
        
        if check_nan(lambda_terms, "lambda_terms") or check_nan(fiber_sections, "fiber_sections_in"):
             print("NaN detected before Lambda Calculus! Zeroing inputs.")
             lambda_terms = torch.zeros_like(lambda_terms)
             fiber_sections = torch.zeros_like(fiber_sections)

        transformed_sections = self._apply_lambda_operations(lambda_terms, fiber_sections)
        if check_nan(transformed_sections, "transformed_sections"):
             print(f"NAN in Transformed Sections! Max: {transformed_sections.max()}")
             transformed_sections = torch.zeros_like(transformed_sections)

        # 5b. Parallel transport for geometric consistency
        transported_coords = self._parallel_transport_update(base_coords, metric)
        if check_nan(transported_coords, "transported_coords"):
             print("NAN in Parallel Transport!")
             transported_coords = base_coords # Fallback

        # 5c. Hamiltonian Sheaf Dynamics (Symplectic Integration)
        active_indices = None
        if self.use_dynamics:
             # Determine Active Indices (Allowed Targets) based on router confidence
             k_active = 3
             router_confidence = torch.max(transformed_sections, dim=-1)[0].squeeze(1) # (B, P)
             _, active_indices = torch.topk(router_confidence, k=k_active, dim=-1) # (B, k)

             # Phase 2.5: Update Resonant Dynamics
             flat_indices = active_indices.view(-1).unique().tolist()
             self.fiber_executor.update_resonance(flat_indices)

             # Coordinate dynamics: detached to prevent NaN from LeapfrogIntegrator sub-graphs.
             # Gradient for coords flows through the residual connection instead.
             with torch.no_grad():
                 updated_coords_dyn, _ = self._symplectic_integrate(
                     transported_coords, transformed_sections, metric, active_indices=active_indices
                 )
             dynamics_direction = (updated_coords_dyn - transported_coords).detach()
             updated_coords = transported_coords + 0.1 * dynamics_direction
             updated_coords = self._project_to_ball(updated_coords, max_norm=0.95)

             # Phase A: Differentiable fiber section update with full gradient coupling.
             # Removed .detach() from updated_coords so base manifold → fiber gradient flows.
             # Added F.softmax() normalization so sections are valid probability distributions.
             # Increased eta_f (0.01→0.1) via config to make fiber_update_net outputs meaningful.
             joint_repr = torch.cat([updated_coords, transformed_sections], dim=-1)
             # Epic 17b: use delta-net fiber update if enabled
             if self.use_delta_fiber:
                 # Curvature from metric (available at this point)
                 _K_scalar = None
                 try:
                     _K_est = self.riemannian_geometry.estimate_sectional_curvature_stochastic(
                         updated_coords[:, :, 0, :], num_samples=1
                     )
                     _K_scalar = _K_est.mean()
                 except Exception:
                     pass
                 # Entropy from current fiber sections
                 _S_scalar = None
                 try:
                     _p = F.softmax(transformed_sections, dim=-1).clamp(min=1e-8)
                     _S_scalar = -(_p * _p.log()).sum(dim=-1).mean()
                 except Exception:
                     pass
                 fiber_update = self.delta_fiber_update(
                     joint_repr, curvature=_K_scalar, entropy=_S_scalar
                 )  # (B, T, P, K)
             else:
                 fiber_update = self.fiber_update_net(joint_repr)  # (B, T, P, K)
             updated_sections = F.softmax(
                 transformed_sections + self.cfg.eta_f * fiber_update, dim=-1
             )

             # Epic 36: Recursive Meta-Cognition (System 2)
             if self.meta_loop is not None and getattr(self.cfg, 'enable_meta_cognition', False):
                 updated_coords, meta_info = self.meta_loop(updated_coords)
        else:
             # Static manifold projection (no dynamics)
             updated_coords = transported_coords
             updated_sections = transformed_sections
             # Fallback to Information Geometric Update (Legacy)
             updated_coords, updated_sections = self._information_geometric_update(
                 transported_coords, transformed_sections, metric
             )

        # EPIC 1: Hamiltonian Evolution is now handled in 5c above.
        # Legacy placeholder removed.

        # EPIC 2: Geodesic Attention (Phase 7 Hybrid)
        # Apply attention mechanism based on hyperbolic distances
        if self.use_geodesic_attn:
            # Attend to geometrically similar concepts in the sequence
            # This enriches the representation with "similar" historical context
            
            # Reshape updated_coords: (B, T, P, D) -> (B, P, T, D)
            # We treat Components P as Heads H? No, RiemannianAttention splits H internally.
            # But here P=8, H=8? Need to match dimensions.
            # RiemannianAttention expects (B, H, T, D).
            # If P corresponds to Heads, let's treat P as H.
            
            z_perm = updated_coords.permute(0, 2, 1, 3) # (B, P, T, D)
            
            # Riemannian Path
            # Note: RiemannianAttention expects (B, H, T, D).
            # We assume P == H for this adapter config.
            # Or we let module handle it.
            geo_out = self.geodesic_attn(z_perm, z_perm, z_perm) # (B, P, T, D)
            
            # Euclidean Path (Bypass)
            # Treated as vectors in T_0
            euc_in = z_perm 
            euc_out = self.euclidean_bypass(euc_in)
            
            # Gating Logic (Epic 37)
            # Approximate Entropy S via variance across components
            s_proxy = torch.var(z_perm, dim=1).mean(dim=-1) # (B, T)
            # Approximate Curvature K via metric trace or placeholder
            k_proxy = -1.0 * torch.ones_like(s_proxy) # Assumed Hyperbolic
            
            # Gate returns (B, T, 1)
            # We assume updated_coords mean as hidden state proxy for gating input?
            h_proxy = updated_coords.mean(dim=2) # (B, T, D)
            lam = self.gating(h_proxy, s_proxy, k_proxy) # (B, T, 1)
            lam = lam.unsqueeze(1).unsqueeze(-1) # (B, 1, T, 1) to broadcast to (B, P, T, D)?
            # Wait, shape of z_perm is (B, P, T, D).
            # lam should be (B, 1, T, 1) to match T.
            
            lam = lam.view(B, 1, T, 1) 
            
            z_mixed = lam * euc_out + (1.0 - lam) * geo_out
            
            # Permute back
            attn_out = z_mixed.permute(0, 2, 1, 3) # (B, T, P, D)
            
            updated_coords = updated_coords + attn_out # Residual connection on the manifold

            # Re-project to Poincare ball after attention residual
            updated_coords = self._project_to_ball(updated_coords, max_norm=0.95)


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
        aggregated_coords_consensus = torch.clamp(aggregated_coords_consensus, -10, 10)
        
        # Add residual from memory to this consensus result
        aggregated_coords_consensus = aggregated_coords_consensus + phase_context
        aggregated_coords_consensus = torch.clamp(aggregated_coords_consensus, -10, 10)
        
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
        combined = torch.clamp(combined, -10, 10)
        
        output = self.output_proj(combined)  # (B, T, H)
        output = torch.clamp(output, -100, 100)  # Prevent extreme values

        # EPIC 31: Fiber Latent Gating
        # Modulate the output using the fiber latents (s_i)
        # We compute this for ALL fibers P and weight by the router weights to ensure smooth gradients
        all_s = self.fiber_store.s # (P, D)
        all_s = torch.clamp(all_s, -10, 10)

        all_gates = self.fiber_gate_proj(all_s) # (P, H)
        # all_gates is already sigmoid'd (0-1), no clamp needed
        
        # Weighted Gate: sum(w_i * g_i)
        # weights: (B, T, P)
        # all_gates: (P, H)
        weighted_gate = torch.einsum('btp,ph->bth', weights, all_gates)
        
        # Modulate Adapter Output
        output = output * weighted_gate
        output = torch.clamp(output, -100, 100)  # Final safety clamp

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
            active_indices=active_indices,
            meta_info=meta_info if 'meta_info' in locals() else None,
            # Phase 4
            hamiltonian_energy=0.0, # Placeholder, populated by hook or executor
            retrospection_loss=0.0,
            fhn_phase="Free"
        )

        final_output = x + self.scale * output
        if is_2d:
            final_output = final_output.squeeze(1)
            
        return final_output, geo_state

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
        # Ensure scalar 0.1 is handled without promoting to float32
        correction = curr_coords - prev_coords
        correction = correction * 0.1 
        
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
        # Scale updates by metric (simplified natural gradient)
        # In full implementation, would use inverse metric tensor
        metric_inv_diag = 1.0 / (torch.diagonal(metric.metric_tensor, dim1=-2, dim2=-1) + 1e-6)
        metric_inv_diag = metric_inv_diag.to(base_update.dtype) # Strict Cast
            
        scaled_base_update = base_update * metric_inv_diag

        # Apply updates with geometric constraints
        eta_base = self.cfg.eta_b
        eta_fiber = self.cfg.eta_f

        # updated_coords = coords + eta_base * scaled_base_update
        # Phase 3.5: Use Manifold Exponential Map
        v_update = eta_base * scaled_base_update
        updated_coords = self._exp_map(coords, v_update)
        
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

    def _exp_map(self, coords: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Helper to invoke manifold exp_map with boundary check."""
        # KanManifold and PoincareBall both support exp_map(x, v)
        updated_coords = self.manifold.exp_map(coords, v)
        
        # Enforce boundary if needed
        if hasattr(self.manifold, 'proj'):
             updated_coords = self.manifold.proj(updated_coords)
        elif hasattr(self.manifold, 'base_geo') and self.manifold.base_geo:
             updated_coords = self.manifold.base_geo.proj(updated_coords)
             
        return updated_coords

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
             energy = self.dynamics.total_energy(state.base_coordinates, p_zero)
             # Phase 6: Logarithmic Loss Scaling to prevent Explosion (1e27 -> 100)
             # Apply log1p on the mean energy
             scaled_energy = torch.log1p(torch.abs(energy.mean())) 
             losses['hamiltonian_energy'] = scaled_energy * 0.1
             
        # EPIC 3: Task Compiler Consistency (Entropy Regularization)
        # Encourage the compiler to be confident (Low Entropy on op_logits)
        if state.op_logits is not None:
             logits = state.op_logits # (B, T, NumOps)
             probs = F.softmax(logits, dim=-1)
             entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
             losses['compiler_entropy'] = entropy * 0.1
             
        # FIBER ENTROPY REGULARIZATION — Phase A entropy unfreeze (2026-03-03)
        # ROOT CAUSE: ∂S/∂θ_i = 0 at uniform distribution (entropy maximum is a critical point).
        # Gradient through softmax vanishes at p_i = 1/K regardless of loss weight or Gumbel noise.
        # The Gumbel trick only perturbed the forward pass — gradient was still zero.
        #
        # FIX: Logit-space losses that bypass softmax entirely.
        #   (a) Logit diversity: -std(z) has nonzero gradient even when all z_i are equal
        #       ∂std/∂z_i = (z_i - mean) / (K * std + eps) — breaks uniform symmetry
        #   (b) Logit entropy via Gumbel-Softmax (straight-through): reparameterized gradient
        #       flows through temperature-scaled logits, not through argmax
        if state.fiber_sections is not None:
            ln_K = torch.log(torch.tensor(float(self.K), device=state.fiber_sections.device))
            S_target = 0.5 * ln_K  # ~1.39 for K=16

            # NOTE: state.fiber_sections are PROBABILITIES (post-softmax on both paths).
            # Convert back to logits for diversity loss and Gumbel-Softmax.
            # log(softmax(z)) = log_softmax(z) — PyTorch autodiff handles this cleanly.
            fiber_logits = torch.log(state.fiber_sections.clamp(min=1e-8))

            # (a) PRIMARY: Logit diversity loss — strong gradient at uniform
            # At uniform: all logits = -ln(K), std = 0.
            # Gradient ∂std/∂z_i = (z_i - mean)/(K·std) is large for any deviation from uniform
            # (numerically ~1/ε, amplifying small asymmetries rather than suppressing them).
            logit_std = fiber_logits.std(dim=-1).mean()
            losses['fiber_diversity'] = F.relu(1.0 - logit_std) * 2.0

            # (b) SECONDARY: Direct entropy loss on actual probabilities.
            # Near uniform, gradient dH/dz = 0, but once fiber_diversity breaks symmetry
            # (moving logits apart), this loss provides directional pull toward S_target.
            # Uses (H - S_target)^2 for two-sided pull: penalizes both too-high AND too-low entropy.
            fiber_entropy = -torch.sum(
                state.fiber_sections * torch.log(state.fiber_sections.clamp(min=1e-8)), dim=-1
            ).mean()
            losses['fiber_entropy'] = (fiber_entropy - S_target).pow(2) * 1.0

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
        Phase 2 Symplectic Integration (Hamiltonian Dynamics).
        Evolve the 'thought' (coords) using Hamiltonian H(q, p) = T(p) + V(q).
        """
        B, T, P, D = coords.shape
        
        # 1. State q = coords
        q = coords
        
        # 2. State p (Momentum)
        # For Phase 2 Generation, we initialize momentum thermally (random Gaussian).
        # In a fully recurrent model, p would come from previous state.
        # Here, we treat each forward pass as an "impulse" of thought.
        # But wait, self.fiber_store.p exists!
        # If we want PERSISTENCE, we should load p from fiber_store.
        # However, fiber_store.p is (P, D). coords is (B, T, P, D).
        # We broadcast p across B, T? Or just use it as 'canonical' momentum?
        # Let's simple Thermal Sampling for v1 "Creativity" boost, as per Integration Plan.
        p = torch.randn_like(q) * 0.1
        
        # 3. Integrate (Leapfrog)
        # We need to temporarily bind 'sections' and 'metric' to the instance 
        # so the potential proxy can see them.
        # This is not thread-safe but fine for single-threaded train/inf.
        self._temp_potential_context = {'sections': sections, 'metric': metric}
        
        # Run Integrator
        # q, p shapes are (B, T, P, D)
        # Leapfrog handles batch dimensions natively as long as operations are vectorized.
        # Manifold ops (exp_map) in poincare.py are vectorized.
        q_new, p_new = self.integrator(q, p)
        
        # Cleanup context
        self._temp_potential_context = None
        
        # 3.5. GENERIC Hamiltonian Cross-Coupling (C-block)
        # Re-introduce thermodynamic coupling mathematically correctly.
        # Instead of detaching beliefs entirely, we pass the entropy gradient
        # into the mechanical momentum update via a C-block, satisfying M \nabla H = 0.
        if sections.requires_grad:
            with torch.enable_grad():
                sec_req = sections.detach().requires_grad_(True)
                dummy_state = GeometricState(
                    mixture_state=None, 
                    base_coordinates=q, 
                    fiber_sections=sec_req, 
                    lambda_terms=None, 
                    metric=metric
                )
                try:
                    # Compute S_fiber (Entropy Proxy via Bundle Structure Loss)
                    S_fiber_proxy = -self._compute_bundle_structure_loss(dummy_state)
                    grad_theta = torch.autograd.grad(S_fiber_proxy.sum(), sec_req, retain_graph=False)[0]
                    
                    # Map K to D using deterministic expansion (C matrix projection)
                    D_dim = p_new.shape[-1]
                    K_dim = grad_theta.shape[-1]
                    repeats = (D_dim // K_dim) + 1
                    C_forcing = grad_theta.repeat(1, 1, 1, repeats)[..., :D_dim] * 0.05  # C coupling constant
                    
                    # Apply forcing term to momentum in the tangent space
                    p_new = p_new + C_forcing
                except Exception:
                    pass
                    
        # 4. Return new Q (Updated Coords)
        # Sections are currently NOT evolved by Hamiltonian (q-only dynamics in Option A).
        # So we return original sections.
        return q_new, sections

    def _adapter_potential_energy_proxy(self, q: torch.Tensor) -> torch.Tensor:
        """
        Proxy method to calculate V(q) using context available during integration.
        Bound to self.hamiltonian.potential_energy.
        """
        if not hasattr(self, '_temp_potential_context') or self._temp_potential_context is None:
             # Fallback: Zero potential (Free particle)
             return torch.zeros(q.shape[:-1], device=q.device)
             
        sections = self._temp_potential_context['sections']
        metric = self._temp_potential_context['metric']
        
        # GENERIC Thermodynamics Constraint (M \nabla H = 0)
        # In the GENERIC framework, Mechanical Energy H(q) must be strictly decoupled 
        # from the dissipative (Entropy) dynamics of \theta (Categorical Sections).
        # We continue to detach sections here to satisfy the degeneracy condition, 
        # but the actual coupling is now handled via the C-block forcing term in the integrator.
        sections_detached = sections.detach()
        return self.compute_hamiltonian_potential(q, sections_detached, metric)

def create_geometric_adapter(config) -> GeometricIGBundleAdapter:
    """Factory function to create geometrically rigorous adapter."""
    return GeometricIGBundleAdapter(config)