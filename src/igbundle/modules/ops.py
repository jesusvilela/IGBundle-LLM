import torch
import torch.nn.functional as F

def bundle_affinity(d_base: torch.Tensor, d_fiber: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Computes bundle affinity matrix A_ij = softmax_j( -alpha*D_base - beta*D_fiber )
    
    Args:
        d_base: (B, T, P, P) Pairwise base divergences (KL Gaussian)
        d_fiber: (B, T, P, P) Pairwise fiber divergences (KL Categorical)
        alpha, beta: Scalar coefficients
        
    Returns:
        A: (B, T, P, P) normalized over last dim (rows sum to 1)
    """
    score = -(alpha * d_base + beta * d_fiber)
    return F.softmax(score, dim=-1)

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
