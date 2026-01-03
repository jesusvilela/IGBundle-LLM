"""
Adaptive Curvature Targeting for IGBundle

This module implements learned curvature targeting that adapts to local data geometry
instead of using fixed hyperbolic targets. The system learns optimal curvature
patterns for different semantic contexts.

Research Hypothesis: Adaptive curvature targeting will outperform fixed targets
by providing geometry that better matches the natural structure of language data.

Author: LLMOS AI Scientist Agent
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

from .riemannian import RiemannianGeometry, RiemannianMetric


class AdaptiveCurvatureTargeting(nn.Module):
    """
    Neural network that learns optimal curvature targets based on local geometry.

    This system replaces fixed curvature targets (-1.0 for hyperbolic) with
    learned targets that adapt to data patterns and semantic contexts.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim
        self.num_components = config.num_components

        # Curvature prediction network
        self.curvature_net = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Single curvature value
        )

        # Context-aware curvature modulation
        self.context_encoder = nn.Sequential(
            nn.Linear(self.dim * 2, 128),  # Current + context
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Hierarchical curvature patterns
        self.hierarchy_net = nn.Sequential(
            nn.Linear(self.dim + 1, 32),  # Position + depth info
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Curvature memory for temporal consistency
        self.curvature_memory = nn.Parameter(
            torch.zeros(self.num_components, 1) - 1.0  # Initialize hyperbolic
        )

        # Learning parameters
        self.curvature_momentum = 0.9
        self.adaptation_rate = 0.1
        self.min_curvature = -5.0
        self.max_curvature = 2.0

    def forward(self, positions: torch.Tensor, context: Optional[torch.Tensor] = None,
                hierarchy_depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive curvature targets for given positions.

        Args:
            positions: (B, T, P, D) - points on manifold
            context: (B, T, D) - optional context information
            hierarchy_depth: (B, T, P) - semantic hierarchy depth

        Returns:
            target_curvatures: (B, T, P) - adaptive curvature targets
        """
        B, T, P, D = positions.shape

        # 1. Local geometry-based curvature
        local_curvatures = self._compute_local_curvatures(positions)

        # 2. Context-modulated curvature
        if context is not None:
            context_curvatures = self._compute_context_curvatures(positions, context)
        else:
            context_curvatures = torch.zeros_like(local_curvatures)

        # 3. Hierarchical curvature adjustment
        if hierarchy_depth is not None:
            hierarchy_curvatures = self._compute_hierarchy_curvatures(positions, hierarchy_depth)
        else:
            hierarchy_curvatures = torch.zeros_like(local_curvatures)

        # 4. Combine curvature components
        combined_curvatures = (
            0.5 * local_curvatures +
            0.3 * context_curvatures +
            0.2 * hierarchy_curvatures
        )

        # 5. Apply memory and smoothing
        smoothed_curvatures = self._apply_temporal_smoothing(combined_curvatures)

        # 6. Clamp to reasonable range
        final_curvatures = torch.clamp(
            smoothed_curvatures,
            min=self.min_curvature,
            max=self.max_curvature
        )

        return final_curvatures

    def _compute_local_curvatures(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute curvature based on local geometry patterns (Vectorized)."""
        B, T, P, D = positions.shape
        
        # 1. Flatten Batch and Time for Batched Computation
        # positions_flat: (N, P, D) where N = B*T
        positions_flat = positions.view(-1, P, D)
        
        # 2. Batched Pairwise Distance Computation
        # distances: (N, P, P)
        distances = torch.cdist(positions_flat, positions_flat)

        # 3. Local Density Estimation using Top-K Neighbors
        # Find k nearest neighbors (excluding self at index 0)
        k = min(3, P - 1)
        if k > 0:
            # topk returns largest, so we negate or look at smallest. 
            # torch.topk with largest=False is standard.
            # values: (N, P, k+1) - includes self (dist=0)
            nearest_distances, _ = torch.topk(distances, k + 1, largest=False, dim=-1)
            
            # Compute mean distance to neighbors (skip the first one which is self/0)
            # mean_dist: (N, P)
            mean_dist = nearest_distances[:, :, 1:].mean(dim=-1)
            
            # Density = Inverse Distance
            local_density = 1.0 / (mean_dist + 1e-6)
            
            # Convert to Curvature: High Density -> High Negative Curvature [-2, 0]
            density_normalized = torch.tanh(local_density)
            curvatures = -2.0 * density_normalized
        else:
            # Fallback for very small P
            curvatures = torch.full((B * T, P), -1.0, device=positions.device)
            
        # 4. Reshape back to (B, T, P)
        return curvatures.view(B, T, P)

        # Neural network refinement
        curvature_refined = []
        for p in range(P):
            pos_p = positions[:, :, p, :]  # (B, T, D)
            curvature_p = self.curvature_net(pos_p).squeeze(-1)  # (B, T)
            curvature_refined.append(curvature_p)

        curvature_refined = torch.stack(curvature_refined, dim=2)  # (B, T, P)

        # Combine analytical and learned components
        return 0.7 * curvatures + 0.3 * curvature_refined

    def _compute_context_curvatures(self, positions: torch.Tensor,
                                   context: torch.Tensor) -> torch.Tensor:
        """Compute curvature based on contextual information."""
        B, T, P, D = positions.shape

        # Expand context for each component
        context_expanded = context.unsqueeze(2).expand(-1, -1, P, -1)  # (B, T, P, D)

        # Combine position and context
        combined_input = torch.cat([positions, context_expanded], dim=-1)  # (B, T, P, 2D)

        # Predict context-aware curvature
        context_curvatures = self.context_encoder(combined_input).squeeze(-1)  # (B, T, P)

        return context_curvatures

    def _compute_hierarchy_curvatures(self, positions: torch.Tensor,
                                    hierarchy_depth: torch.Tensor) -> torch.Tensor:
        """Compute curvature based on semantic hierarchy depth."""
        B, T, P, D = positions.shape

        # Normalize hierarchy depth
        depth_normalized = F.tanh(hierarchy_depth)  # (B, T, P)

        # Combine position and depth information
        depth_expanded = depth_normalized.unsqueeze(-1)  # (B, T, P, 1)
        hierarchy_input = torch.cat([positions, depth_expanded], dim=-1)  # (B, T, P, D+1)

        # Predict hierarchy-aware curvature
        hierarchy_curvatures = self.hierarchy_net(hierarchy_input).squeeze(-1)  # (B, T, P)

        return hierarchy_curvatures

    def _apply_temporal_smoothing(self, curvatures: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing using momentum-based memory."""
        B, T, P = curvatures.shape

        # Update memory with current curvatures (exponential moving average)
        current_mean = curvatures.mean(dim=(0, 1))  # (P,)

        # Update memory
        self.curvature_memory.data = (
            self.curvature_momentum * self.curvature_memory.data +
            (1 - self.curvature_momentum) * current_mean.unsqueeze(-1)
        )

        # Apply memory-based smoothing
        memory_influence = self.curvature_memory.squeeze(-1)  # (P,)
        memory_expanded = memory_influence.view(1, 1, P).expand(B, T, P)

        # Blend current and memory
        smoothed = (
            (1 - self.adaptation_rate) * curvatures +
            self.adaptation_rate * memory_expanded
        )

        return smoothed


class DynamicCurvatureScheduler(nn.Module):
    """
    Dynamic curvature scheduling that adapts the rate of curvature change
    based on training progress and geometric learning dynamics.
    """

    def __init__(self, config):
        super().__init__()

        # Scheduling parameters
        self.initial_curvature = getattr(config, 'initial_target_curvature', 0.0)
        self.final_curvature = getattr(config, 'final_target_curvature', -1.0)

        # Adaptive scheduling network
        self.schedule_net = nn.Sequential(
            nn.Linear(3, 32),  # progress, loss_reduction, curvature_alignment
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, training_progress: float, loss_reduction: float,
                curvature_alignment: float) -> float:
        """
        Compute adaptive curvature schedule based on training dynamics.

        Args:
            training_progress: Fraction of training completed [0, 1]
            loss_reduction: Recent loss reduction rate
            curvature_alignment: How well current geometry matches targets

        Returns:
            scheduled_curvature: Current target curvature
        """

        # Create input features
        features = torch.tensor([
            training_progress,
            loss_reduction,
            curvature_alignment
        ], dtype=torch.float32)

        # Predict schedule modification
        schedule_modifier = self.schedule_net(features.unsqueeze(0)).squeeze()

        # Base exponential schedule
        base_schedule = self.initial_curvature + (
            self.final_curvature - self.initial_curvature
        ) * (1 - math.exp(-3 * training_progress))

        # Apply learned modification
        final_schedule = base_schedule + 0.5 * schedule_modifier

        return float(final_schedule)


def adaptive_curvature_loss(adaptive_targeting: AdaptiveCurvatureTargeting,
                          geometry: RiemannianGeometry,
                          positions: torch.Tensor,
                          context: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute curvature loss using adaptive targets instead of fixed targets.

    Args:
        adaptive_targeting: AdaptiveCurvatureTargeting module
        geometry: RiemannianGeometry instance
        positions: (B, T, P, D) - points on manifold
        context: Optional context information

    Returns:
        loss: Adaptive curvature regularization loss
    """
    B, T, P, D = positions.shape

    # Get adaptive curvature targets
    target_curvatures = adaptive_targeting(positions, context)  # (B, T, P)

    # Generate random orthonormal tangent vectors
    u = torch.randn_like(positions)
    v = torch.randn_like(positions)

    # Gram-Schmidt orthogonalization
    metric = geometry.get_metric(positions)
    u_norm = u / torch.sqrt(geometry.inner_product(u, u, metric) + 1e-8).unsqueeze(-1)

    u_v_inner = geometry.inner_product(u_norm, v, metric)
    v_orth = v - u_v_inner.unsqueeze(-1) * u_norm
    v_norm = v_orth / torch.sqrt(geometry.inner_product(v_orth, v_orth, metric) + 1e-8).unsqueeze(-1)

    # Compute sectional curvature
    actual_curvatures = geometry.sectional_curvature(positions, u_norm, v_norm)  # (B, T, P)

    # Adaptive loss with per-point targets
    curvature_loss = F.mse_loss(actual_curvatures, target_curvatures)

    return curvature_loss


def create_adaptive_curvature_system(config) -> Tuple[AdaptiveCurvatureTargeting, DynamicCurvatureScheduler]:
    """Factory function to create adaptive curvature system."""
    adaptive_targeting = AdaptiveCurvatureTargeting(config)
    dynamic_scheduler = DynamicCurvatureScheduler(config)

    return adaptive_targeting, dynamic_scheduler