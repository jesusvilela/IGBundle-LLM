import torch
import torch.nn.functional as F

def compute_poincare_distance(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Computes pairwise Poincare distance between sets of vectors.
    x: (B, T, P, D)
    y: (B, T, P, D)
    Returns: (B, T, P, P) pairwise distance matrix
    """
    # Mobius addition-based distance or isometric invariant definition
    # d(u,v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    
    # We need pairwise computation. 
    # x_i: (B, T, P, 1, D)
    # y_j: (B, T, 1, P, D)
    x_i = x.unsqueeze(3)
    y_j = y.unsqueeze(2)
    
    # Squared Euclidean distance in the ambient space (ball)
    sq_euc_dist = torch.sum((x_i - y_j).pow(2), dim=-1)
    
    # Norms
    x_sq_norm = torch.sum(x_i.pow(2), dim=-1)
    y_sq_norm = torch.sum(y_j.pow(2), dim=-1)
    
    # Clip norms to ensure stability (stay inside ball)
    x_sq_norm = torch.clamp(x_sq_norm, max=1.0-epsilon)
    y_sq_norm = torch.clamp(y_sq_norm, max=1.0-epsilon)
    
    # Argument for arccosh
    alpha_num_stab = 1e-7 # Numerical stabilizer
    val = 1.0 + 2.0 * sq_euc_dist / ((1.0 - x_sq_norm) * (1.0 - y_sq_norm) + alpha_num_stab)
    
    # Arccosh
    dist = torch.acosh(torch.clamp(val, min=1.0 + epsilon))
    return dist

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


def compute_affinity_matrix(means: torch.Tensor, log_sigmas: torch.Tensor, fiber_divergences: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """
    Compute mixing weights A using Riemannian Geodesic Distance.
    
    Args:
        means: (B, T, P, D) Mean vectors for base manifold.
        log_sigmas: (B, T, P, D) Log standard deviations for scaling.
        fiber_divergences: (B, T, P, P) Pairwise fiber divergences (e.g., KL Categorical).
        alpha, beta: Scalar coefficients for combining base and fiber energies.
        
    Returns:
        A: (B, T, P, P) normalized over last dim (rows sum to 1)
    """
    # 1. Base Manifold: Hyperbolic (Poincare Ball)
    # Project Euclidean means to Poincare Ball via Tanh
    means_hyp = torch.tanh(means) 
    
    # Compute Geodesic Distances
    # We interpret 'log_sigmas' not as variance, but as Inverse Curvature / Temperature
    # Higher sigma = flatter/wider = larger neighborhood = higher temperature
    D_base = compute_poincare_distance(means_hyp, means_hyp)
    
    # 2. Fiber: Categorical Divergence (Jensen-Shannon or KL)
    # We keep the "Fiber" logic as categorical distributions
    # Assuming fiber_divergences is already the pairwise divergence matrix
    D_fiber = fiber_divergences # This directly uses the input, assuming it's the pre-computed divergence.
                                # If fiber_divergences were logits, we'd call compute_pairwise_kl_categorical(fiber_divergences)
    
    # 3. Combine
    # Use sigma as temperature scaling for the base distance
    # sigma = exp(log_sigma)
    sigmas = torch.exp(log_sigmas).mean(dim=-1, keepdim=True) # Average sigma per component (B,T,P,1)
    
    # Adaptive Temperature: T_ij = sqrt(sigma_i * sigma_j)
    sigmas_i = sigmas.unsqueeze(3)
    sigmas_j = sigmas.unsqueeze(2)
    T_ij = torch.sqrt(sigmas_i * sigmas_j) + 1e-6
    
    # Energy = alpha * (D_base / T_ij) + beta * D_fiber
    # We effectively scale the hyperbolic distance by the learned uncertainty
    energy = alpha * (D_base / T_ij) + beta * D_fiber
    
    # Affinity = Softmax(-Energy)
    A = F.softmax(-energy, dim=-1)
    
    return A

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
