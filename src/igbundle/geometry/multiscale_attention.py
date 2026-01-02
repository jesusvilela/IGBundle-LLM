"""
Multi-Scale Geometric Attention for IGBundle

This module implements geometric attention mechanisms that operate at multiple
scales simultaneously, capturing both local geometric structure and global
manifold patterns for enhanced semantic representation.

Research Hypothesis: Multi-scale geometric attention will improve compositional
understanding by maintaining geometric consistency across different resolution levels.

Author: LLMOS AI Scientist Agent
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import math

from .riemannian import RiemannianGeometry, RiemannianMetric


class MultiScaleMetric(nn.Module):
    """
    Riemannian metric that operates at multiple scales simultaneously.

    This implements scale-aware metric tensors that capture geometric
    structure at different resolution levels.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim
        self.num_scales = getattr(config, 'num_geometric_scales', 3)
        self.num_components = config.num_components

        # Scale-specific metric parameters
        self.scale_metrics = nn.ModuleList([
            nn.Parameter(torch.randn(self.num_components, self.dim, self.dim) * 0.01)
            for _ in range(self.num_scales)
        ])

        # Scale fusion network
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.num_scales, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_scales),
            nn.Softmax(dim=-1)
        )

        # Scale encoding
        self.scale_encodings = nn.Parameter(
            torch.randn(self.num_scales, 16) * 0.1
        )

    def forward(self, positions: torch.Tensor, scale_weights: Optional[torch.Tensor] = None) -> List[RiemannianMetric]:
        """
        Compute multi-scale metric tensors.

        Args:
            positions: (B, T, P, D) - points on manifold
            scale_weights: (B, T, P, num_scales) - optional scale attention weights

        Returns:
            metrics: List of RiemannianMetric objects, one per scale
        """
        B, T, P, D = positions.shape

        if scale_weights is None:
            # Compute automatic scale weights based on position
            scale_weights = self._compute_scale_attention(positions)

        metrics = []

        for scale_idx in range(self.num_scales):
            # Get scale-specific metric parameters
            L_scale = torch.tril(self.scale_metrics[scale_idx])  # (P, D, D)

            # Compute metric as g = L * L^T
            metric_scale = torch.matmul(L_scale, L_scale.transpose(-1, -2))  # (P, D, D)

            # Apply scale-specific modulation
            scale_weight = scale_weights[:, :, :, scale_idx].unsqueeze(-1).unsqueeze(-1)  # (B, T, P, 1, 1)
            metric_scale = metric_scale.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            metric_scale = metric_scale * scale_weight

            metrics.append(RiemannianMetric(metric_scale))

        return metrics

    def _compute_scale_attention(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute attention weights across scales."""
        B, T, P, D = positions.shape

        # Compute local geometric features for scale selection
        position_variance = torch.var(positions, dim=-1, keepdim=True)  # (B, T, P, 1)
        position_magnitude = torch.norm(positions, dim=-1, keepdim=True)  # (B, T, P, 1)

        # Create scale features
        scale_features = torch.cat([
            position_variance.expand(-1, -1, -1, self.num_scales),
            position_magnitude.expand(-1, -1, -1, self.num_scales),
            self.scale_encodings.view(1, 1, 1, self.num_scales, -1).expand(B, T, P, -1, -1).sum(dim=-1)
        ], dim=-1)  # (B, T, P, num_scales * 3)

        # Compute scale attention
        scale_weights = []
        for p in range(P):
            features_p = scale_features[:, :, p, :]  # (B, T, num_scales * 3)
            weights_p = self.scale_fusion(features_p[:, :, :self.num_scales])  # (B, T, num_scales)
            scale_weights.append(weights_p)

        scale_weights = torch.stack(scale_weights, dim=2)  # (B, T, P, num_scales)

        return scale_weights


class CrossScaleAttention(nn.Module):
    """
    Attention mechanism that operates across different geometric scales.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim
        self.num_scales = getattr(config, 'num_geometric_scales', 3)
        self.num_heads = getattr(config, 'num_geometric_heads', 4)

        # Multi-head attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.dim,
                num_heads=self.num_heads,
                batch_first=True
            )
            for _ in range(self.num_scales)
        ])

        # Cross-scale fusion
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(self.dim * self.num_scales, self.dim * 2),
            nn.ReLU(),
            nn.Linear(self.dim * 2, self.dim),
            nn.LayerNorm(self.dim)
        )

        # Scale-aware position encoding
        self.scale_pos_encoder = nn.Sequential(
            nn.Linear(self.dim + 1, 64),  # position + scale_id
            nn.ReLU(),
            nn.Linear(64, self.dim)
        )

    def forward(self, positions: torch.Tensor, multiscale_metrics: List[RiemannianMetric]) -> torch.Tensor:
        """
        Apply cross-scale geometric attention.

        Args:
            positions: (B, T, P, D) - points on manifold
            multiscale_metrics: List of metrics for different scales

        Returns:
            attended_positions: (B, T, P, D) - scale-attended positions
        """
        B, T, P, D = positions.shape

        # Process each scale
        scale_outputs = []

        for scale_idx, metric in enumerate(multiscale_metrics):
            # Add scale information to positions
            scale_id = torch.full((B, T, P, 1), scale_idx / self.num_scales, device=positions.device)
            pos_with_scale = torch.cat([positions, scale_id], dim=-1)  # (B, T, P, D+1)

            # Scale-aware position encoding
            pos_encoded = self.scale_pos_encoder(pos_with_scale)  # (B, T, P, D)

            # Reshape for attention (combine B, T for batch processing)
            pos_flat = pos_encoded.view(B * T, P, D)

            # Apply self-attention at this scale
            attended, _ = self.scale_attentions[scale_idx](
                pos_flat, pos_flat, pos_flat
            )  # (B*T, P, D)

            # Reshape back
            attended = attended.view(B, T, P, D)
            scale_outputs.append(attended)

        # Fuse across scales
        scale_concatenated = torch.cat(scale_outputs, dim=-1)  # (B, T, P, D * num_scales)
        fused_output = self.cross_scale_fusion(scale_concatenated)  # (B, T, P, D)

        return fused_output


class MultiScaleParallelTransport(nn.Module):
    """
    Parallel transport that maintains consistency across multiple geometric scales.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim
        self.num_scales = getattr(config, 'num_geometric_scales', 3)

        # Transport networks for each scale
        self.transport_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim * 2, 64),  # vector + path
                nn.Tanh(),
                nn.Linear(64, self.dim)
            )
            for _ in range(self.num_scales)
        ])

        # Cross-scale consistency network
        self.consistency_net = nn.Sequential(
            nn.Linear(self.dim * self.num_scales, 128),
            nn.ReLU(),
            nn.Linear(128, self.dim),
            nn.Tanh()
        )

    def forward(self, vectors: torch.Tensor, paths: torch.Tensor,
                multiscale_metrics: List[RiemannianMetric]) -> torch.Tensor:
        """
        Transport vectors along paths maintaining cross-scale consistency.

        Args:
            vectors: (B, T, P, D) - vectors to transport
            paths: (B, T, P, N, D) - path points
            multiscale_metrics: Metrics for different scales

        Returns:
            transported: (B, T, P, D) - transported vectors
        """
        B, T, P, N, D = paths.shape

        # Transport at each scale
        scale_transports = []

        for scale_idx, metric in enumerate(multiscale_metrics):
            # Simple transport approximation (could be enhanced with proper geodesics)
            path_diffs = paths[:, :, :, 1:, :] - paths[:, :, :, :-1, :]  # (B, T, P, N-1, D)
            path_summary = path_diffs.mean(dim=3)  # (B, T, P, D)

            # Combine vector and path information
            transport_input = torch.cat([vectors, path_summary], dim=-1)  # (B, T, P, 2D)

            # Apply scale-specific transport
            transported_scale = self.transport_nets[scale_idx](transport_input)  # (B, T, P, D)
            scale_transports.append(transported_scale)

        # Enforce cross-scale consistency
        scale_concat = torch.cat(scale_transports, dim=-1)  # (B, T, P, D * num_scales)
        consistency_correction = self.consistency_net(scale_concat)  # (B, T, P, D)

        # Combine with weighted average
        scale_weights = torch.softmax(
            torch.randn(self.num_scales, device=vectors.device), dim=0
        )

        weighted_transport = sum(
            w * transport for w, transport in zip(scale_weights, scale_transports)
        )

        # Apply consistency correction
        final_transport = weighted_transport + 0.1 * consistency_correction

        return final_transport


class MultiScaleGeometricAdapter(nn.Module):
    """
    Complete multi-scale geometric adapter that integrates all components.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.latent_dim
        self.num_scales = getattr(config, 'num_geometric_scales', 3)

        # Core multi-scale components
        self.multiscale_metric = MultiScaleMetric(config)
        self.cross_scale_attention = CrossScaleAttention(config)
        self.multiscale_transport = MultiScaleParallelTransport(config)

        # Scale selection network
        self.scale_selector = nn.Sequential(
            nn.Linear(self.dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_scales),
            nn.Softmax(dim=-1)
        )

        # Output projection
        self.output_projection = nn.Linear(self.dim, self.dim)

    def forward(self, positions: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Multi-scale geometric processing.

        Args:
            positions: (B, T, P, D) - input positions
            context: Optional context information

        Returns:
            processed_positions: (B, T, P, D) - multi-scale processed positions
            scale_info: Dict with scale-related information
        """
        B, T, P, D = positions.shape

        # 1. Compute multi-scale metrics
        multiscale_metrics = self.multiscale_metric(positions)

        # 2. Apply cross-scale attention
        attended_positions = self.cross_scale_attention(positions, multiscale_metrics)

        # 3. Create paths for transport (simple case: identity paths)
        paths = positions.unsqueeze(3)  # (B, T, P, 1, D) - single point paths

        # 4. Apply multi-scale parallel transport
        transported_positions = self.multiscale_transport(
            attended_positions, paths, multiscale_metrics
        )

        # 5. Final projection
        output_positions = self.output_projection(transported_positions)

        # 6. Collect scale information for analysis
        scale_weights = self.multiscale_metric._compute_scale_attention(positions)

        scale_info = {
            'scale_weights': scale_weights,
            'num_scales': self.num_scales,
            'scale_diversity': torch.std(scale_weights, dim=-1).mean(),
            'dominant_scale': torch.argmax(scale_weights.mean(dim=(0, 1, 2)), dim=-1)
        }

        return output_positions, scale_info


def multiscale_geometric_loss(multiscale_adapter: MultiScaleGeometricAdapter,
                            positions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute multi-scale geometric regularization losses.

    Args:
        multiscale_adapter: MultiScaleGeometricAdapter instance
        positions: (B, T, P, D) - positions on manifold

    Returns:
        losses: Dict of different loss components
    """
    processed_positions, scale_info = multiscale_adapter(positions)

    losses = {}

    # Scale diversity loss (encourage using multiple scales)
    scale_weights = scale_info['scale_weights']  # (B, T, P, num_scales)
    scale_entropy = -torch.sum(scale_weights * torch.log(scale_weights + 1e-8), dim=-1)
    losses['scale_diversity'] = -scale_entropy.mean()  # Maximize entropy

    # Cross-scale consistency loss
    multiscale_metrics = multiscale_adapter.multiscale_metric(positions)
    consistency_loss = 0.0

    for i in range(len(multiscale_metrics)):
        for j in range(i + 1, len(multiscale_metrics)):
            # Compare metric tensors across scales
            metric_i = multiscale_metrics[i].metric_tensor.mean(dim=(0, 1))  # (P, D, D)
            metric_j = multiscale_metrics[j].metric_tensor.mean(dim=(0, 1))  # (P, D, D)

            # Encourage smooth transitions between scales
            consistency_loss += F.mse_loss(metric_i, metric_j) / (len(multiscale_metrics) * (len(multiscale_metrics) - 1) / 2)

    losses['cross_scale_consistency'] = consistency_loss

    # Position preservation loss
    losses['position_preservation'] = F.mse_loss(processed_positions, positions) * 0.1

    return losses


def create_multiscale_geometric_system(config) -> MultiScaleGeometricAdapter:
    """Factory function to create multi-scale geometric system."""
    return MultiScaleGeometricAdapter(config)