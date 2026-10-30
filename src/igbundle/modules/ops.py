import torch
import torch.nn.functional as F


def project_to_poincare_ball(x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
    """Radially clip vectors to a closed ball strictly inside the unit ball."""
    if not 0 < max_norm < 1:
        raise ValueError("max_norm must be strictly between zero and one")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Poincare coordinates must be finite floating-point values")
    # Accumulate low-precision norms in float32; preserve float64 for diagnostics.
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    norm = torch.linalg.vector_norm(work, dim=-1, keepdim=True)
    if not torch.isfinite(norm).all():
        raise ValueError("Coordinate norm overflow during Poincare projection")
    return work * (max_norm / norm.clamp_min(max_norm))


def compute_poincare_distance(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Unit-curvature Poincare distances for x (..., P, D), y (..., Q, D).

    Inputs must lie strictly inside the unit ball; projection is explicit.
    Half/bfloat16 inputs are evaluated in float32. At coincident points the
    norm's zero subgradient is used. epsilon is retained for API compatibility,
    not added to the distance or the metric denominator.
    """
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be strictly between zero and one")
    if x.ndim < 2 or y.ndim < 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError("Expected point sets with matching coordinate dimensions")
    for points in (x, y):
        if not points.is_floating_point() or not torch.isfinite(points).all():
            raise ValueError("Poincare coordinates must be finite floating-point values")
    x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
    x_sq = x.square().sum(dim=-1)
    y_sq = y.square().sum(dim=-1)
    if (x_sq >= 1).any() or (y_sq >= 1).any():
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    numerator = torch.linalg.vector_norm(x.unsqueeze(-2) - y.unsqueeze(-3), dim=-1)
    denominator = torch.sqrt((1 - x_sq).unsqueeze(-1) * (1 - y_sq).unsqueeze(-2))
    # acosh(1 + 2r^2) = 2 asinh(r), without cancellation or a biased diagonal.
    return 2 * torch.asinh(numerator / denominator)

# Assuming compute_pairwise_kl_categorical is defined elsewhere or needs to be added.
# For this edit, we'll define a placeholder that mimics the original d_fiber input.
def compute_pairwise_kl_categorical(fiber_logits: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for computing pairwise KL divergence for categorical distributions.
    In the original bundle_affinity, d_fiber was directly provided.
    This function would typically compute KL(P_i || P_j) for all i, j.
    
    For now, we'll assume fiber_logits is already the pairwise divergence
    or that the actual implementation will be provided.
    If fiber_logits is meant to be actual logits, this function would need
    to compute the pairwise KL from them.
    
    Given the original `d_fiber: (B, T, P, P)` input to `bundle_affinity`,
    we'll assume `fiber_logits` here is effectively that pre-computed divergence.
    """
    # This is a placeholder. A real implementation would compute KL from logits.
    # For example, if fiber_logits were (B, T, P, K) where K is num categories:
    # log_probs = F.log_softmax(fiber_logits, dim=-1)
    # kl_divs = []
    # for i in range(P):
    #     for j in range(P):
    #         kl_divs.append(F.kl_div(log_probs[:,:,i,:], log_probs[:,:,j,:], reduction='none').sum(dim=-1))
    # return torch.stack(kl_divs, dim=-1).reshape(B, T, P, P)
    
    # If fiber_logits is already the divergence matrix, just return it.
    # This aligns with the original `d_fiber` input to `bundle_affinity`.
    return fiber_logits


from .kl import kl_diag_gauss

def compute_affinity_matrix(means: torch.Tensor, log_sigmas: torch.Tensor, fiber_divergences: torch.Tensor, alpha: float = 1.0, beta: float = 1.0, geometry: str = 'riemannian') -> torch.Tensor:
    """
    Compute mixing weights A using Riemannian (Poincare) or Euclidean (KL) geometry.
    
    Args:
        means: (B, T, P, D) Mean vectors.
        log_sigmas: (B, T, P, D) Log stat deviations (or inverse curvature).
        fiber_divergences: (B, T, P, P) Pairwise fiber divergences.
        alpha, beta: Scalar coefficients.
        geometry: 'riemannian' or 'euclidean'.
        
    Returns:
        A: (B, T, P, P) normalized over last dim.
    """
    
    if geometry not in ('euclidean', 'riemannian'):
        raise ValueError(f"Unknown affinity geometry: {geometry!r}")
    if geometry == 'euclidean':
        # 1. Base Manifold: Euclidean (Gaussian KL)
        # We assume means are in Euclidean space (no projection)
        m_i = means.unsqueeze(3)
        m_j = means.unsqueeze(2)
        ls_i = log_sigmas.unsqueeze(3)
        ls_j = log_sigmas.unsqueeze(2)
        
        # D_base = KL(N_i || N_j)
        D_base = kl_diag_gauss(m_i, ls_i, m_j, ls_j)
        
        # 2. Fiber: Categorical Divergence
        D_fiber = fiber_divergences
        
        # 3. Combine (Standard Energy)
        # Energy = alpha * D_base + beta * D_fiber
        energy = alpha * D_base + beta * D_fiber
        
    else: # riemannian (default)
        # 1. Base Manifold: Hyperbolic (Poincare Ball)
        means_hyp = project_to_poincare_ball(means)
        
        # Compute Geodesic Distances
        # We interpret 'log_sigmas' not as variance, but as Inverse Curvature / Temperature
        D_base = compute_poincare_distance(means_hyp, means_hyp)
        
        # 2. Fiber: Categorical Divergence
        D_fiber = fiber_divergences
        
        # 3. Combine
        # Use sigma as temperature scaling for the base distance
        sigmas = torch.exp(log_sigmas).mean(dim=-1, keepdim=True) # (B,T,P,1)
        
        # Adaptive Temperature: T_ij = sqrt(sigma_i * sigma_j)
        sigmas_i = sigmas.unsqueeze(3)
        sigmas_j = sigmas.unsqueeze(2)
        T_ij = (torch.sqrt(sigmas_i * sigmas_j) + 1e-6).squeeze(-1) # (B, T, P, P)
        
        # Debug shapes
        # print(f"DEBUG: D_base={D_base.shape}, T_ij={T_ij.shape}, D_fiber={D_fiber.shape}")
        
        # Energy = alpha * (D_base / T_ij) + beta * D_fiber
        energy = alpha * (D_base / T_ij) + beta * D_fiber
    
    # Affinity = Softmax(-Energy)
    A = F.softmax(-energy, dim=-1)
    
    return A.to(means.dtype)

def mix_messages(affinity: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes message aggregation: h_i = sum_j A_ij * x_j
    
    Args:
        affinity: (B, T, P, P)
        x: (B, T, P, D_feat)
        
    Returns:
        mixed: (B, T, P, D_feat)
    """
    # Matrix multiplication over the P dims
    # x shape needs P as second last for matmul if we view it as (B*T, P, D)
    # But PyTorch matmul broadcasts.
    # A: (..., P_out, P_in)
    # x: (..., P_in, D)
    # out: (..., P_out, D)
    
    return torch.matmul(affinity, x)
