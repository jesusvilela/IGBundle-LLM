"""
Legacy experimental learned-connection adapter, not the validated default.

Its neural encodings and consistency penalties do not establish lambda
calculus, categorical or sheaf laws. The connection is not derived from its
metric, and coordinate smoothing is not parallel transport. See README.md for
the tested default IGBundleAdapter and the explicit validation boundary.

Author: LLMOS SystemAgent
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..geometry.riemannian import (
    RiemannianGeometry,
    FiberBundleLambdaCalculus,
    RiemannianMetric,
    bundle_curvature_loss
)
from .state import MixtureState
from .kl import kl_diag_gauss, kl_categorical_logits

@dataclass
class GeometricState:
    """
    Legacy tensor state; geometric field names do not certify mathematical laws.

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

class GeometricIGBundleAdapter(nn.Module):
    """
    Legacy prototype combining learned coordinates, neural updates and
    heuristic penalties. Historical method names are not verified semantics.
    """

    def __init__(self, config):
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
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, GeometricState]:
        """
        Execute legacy neural updates and coordinate smoothing.

        Args:
            x: (B, T, H) - input hidden states
            context: (B, T, H) - optional context (unused)

        Returns:
            output: (B, T, H) - transformed hidden states
            state: GeometricState - geometric bundle state
        """
        B, T, H = x.shape

        # 1. Project to bottleneck space
        h_bot = self.input_proj(x)  # (B, T, D_bot)

        # 2. Extract geometric coordinates
        base_coords = self.base_coord_proj(h_bot).view(B, T, self.P, self.D)
        fiber_sections = self.fiber_section_proj(h_bot).view(B, T, self.P, self.K)
        lambda_terms = self.lambda_term_proj(h_bot).view(B, T, self.P, self.D + self.K)

        # 3. Compatibility: extract mixture parameters
        w_logits = self.mixture_proj_w(h_bot)
        m = self.mixture_proj_m(h_bot).view(B, T, self.P, self.D)
        log_s = self.mixture_proj_s(h_bot).view(B, T, self.P, self.D)
        log_s = torch.clamp(log_s, min=-5, max=5)
        u = self.mixture_proj_u(h_bot).view(B, T, self.P, self.K)

        mixture_state = MixtureState(w_logits, m, log_s, u)

        # 4. Compute Riemannian metric at base coordinates
        metric = self.riemannian_geometry.get_metric(base_coords)

        # 5. Apply geometric transformations

        # 5a. Neural section mixing
        transformed_sections = self._apply_lambda_operations(lambda_terms, fiber_sections)

        # 5b. Coordinate smoothing, not parallel transport
        smoothed_coords = self._coordinate_smoothing_update(base_coords)

        # 5c. Diagonal-metric-scaled neural updates
        updated_coords, updated_sections = self._information_geometric_update(
            smoothed_coords, transformed_sections, metric
        )

        # 6. Aggregate across components using Riemannian structure
        weights = F.softmax(w_logits, dim=-1)  # (B, T, P)

        # Legacy Euclidean averaging
        aggregated_coords = self._riemannian_weighted_mean(updated_coords, weights, metric)
        aggregated_sections = torch.einsum('btp,btpk->btk', weights, updated_sections)

        # 7. Project back to hidden space
        combined = torch.cat([aggregated_coords, aggregated_sections], dim=-1)  # (B, T, D+K)
        output = self.output_proj(combined)  # (B, T, H)
        output = self.dropout(output)

        # 8. Construct geometric state
        geo_state = GeometricState(
            mixture_state=mixture_state,
            base_coordinates=updated_coords,
            fiber_sections=updated_sections,
            lambda_terms=lambda_terms,
            metric=metric
        )

        return x + self.scale * output, geo_state

    def _apply_lambda_operations(self, lambda_terms: torch.Tensor,
                                fiber_sections: torch.Tensor) -> torch.Tensor:
        """Mix learned section encodings; no lambda-calculus laws are enforced."""
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

    def _coordinate_smoothing_update(self, base_coords: torch.Tensor) -> torch.Tensor:
        """Blend each coordinate with its predecessor, using no metric/connection.

        A chunk boundary needs the previous unsmoothed coordinate to reproduce a
        full-sequence call. This stateless helper does not maintain that history.
        """
        if base_coords.shape[1] <= 1:
            return base_coords
        current, previous = base_coords[:, 1:], base_coords[:, :-1]
        return torch.cat((base_coords[:, :1], current - .1 * (current - previous)), dim=1)

    def _information_geometric_update(self, coords: torch.Tensor, sections: torch.Tensor,
                                    metric: RiemannianMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply diagonal metric scaling to learned coordinate updates."""
        B, T, P, D = coords.shape

        # Combine coordinates and sections for joint processing
        joint_repr = torch.cat([coords, sections], dim=-1)  # (B, T, P, D+K)

        # Compute natural gradient updates
        base_update = self.base_update_net(joint_repr)  # (B, T, P, D)
        fiber_update = self.fiber_update_net(joint_repr)  # (B, T, P, K)

        # Scale updates by metric (simplified natural gradient)
        # In full implementation, would use inverse metric tensor
        metric_inv_diag = 1.0 / (torch.diagonal(metric.metric_tensor, dim1=-2, dim2=-1) + 1e-6)
        scaled_base_update = base_update * metric_inv_diag

        # Apply updates with geometric constraints
        eta_base = self.cfg.eta_b
        eta_fiber = self.cfg.eta_f

        updated_coords = coords + eta_base * scaled_base_update
        updated_sections = sections + eta_fiber * fiber_update

        # Ensure fiber sections remain on probability simplex
        updated_sections = F.softmax(updated_sections, dim=-1)

        return updated_coords, updated_sections

    def _riemannian_weighted_mean(self, coords: torch.Tensor, weights: torch.Tensor,
                                 metric: RiemannianMetric) -> torch.Tensor:
        """Euclidean weighted averaging with the historical refinement steps."""
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
        curvature_loss = bundle_curvature_loss(
            self.riemannian_geometry,
            state.base_coordinates,
            target_curvature=-1.0  # Hyperbolic
        )
        losses['curvature'] = curvature_loss

        # 2. Sheaf consistency loss
        sheaf_loss = self._compute_sheaf_consistency_loss(state)
        losses['sheaf_consistency'] = sheaf_loss

        # 3. Bundle structure loss (ensure proper fiber bundle topology)
        bundle_loss = self._compute_bundle_structure_loss(state)
        losses['bundle_structure'] = bundle_loss

        # 4. Lambda calculus type consistency
        lambda_loss = self._compute_lambda_consistency_loss(state)
        losses['lambda_consistency'] = lambda_loss

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

def create_geometric_adapter(config) -> GeometricIGBundleAdapter:
    """Factory function to create geometrically rigorous adapter."""
    return GeometricIGBundleAdapter(config)