import torch
import torch.nn.functional as F

def kl_diag_gauss(m1: torch.Tensor, log_s1: torch.Tensor, m2: torch.Tensor, log_s2: torch.Tensor) -> torch.Tensor:
    """
    Computes KL(N(m1, s1) || N(m2, s2)) for diagonal Gaussians.
    Summed over the last dimension (D_lat).
    
    Formula:
    KL = 0.5 * sum [ (s1^2 / s2^2) + (m2 - m1)^2 / s2^2 - 1 + 2(log_s2 - log_s1) ]
    
    Args:
        m1, log_s1: Parameters of first distribution.
        m2, log_s2: Parameters of second distribution.
        Shapes must be broadcastable, e.g. (B, T, P, 1, D) vs (B, T, 1, P, D).
        
    Returns:
        KL divergence tensor of shape broadcast(m1, m2) excluding the last dim.
    """
    # Variance
    v1 = torch.exp(2 * log_s1)
    v2 = torch.exp(2 * log_s2)
    
    # Term 1: log(v2/v1) -> 2*(log_s2 - log_s1)
    term_log = 2 * (log_s2 - log_s1)
    
    # Term 2: (v1 + (m1-m2)^2) / v2
    diff = m1 - m2
    term_trace_qc = (v1 + diff.pow(2)) / v2
    
    # KL sum over D
    # 0.5 * sum( term_trace_qc - 1 - term_log )
    # = 0.5 * sum( term_trace_qc - 1 + term_log ) -- wait, formula check
    # KL = 0.5 * ( tr(inv(S2)S1) + (m2-m1)^T inv(S2) (m2-m1) - k + ln(det(S2)/det(S1)) )
    # For diagonal: sum( (v1_i + (m1_i-m2_i)^2)/v2_i - 1 + ln(v2_i) - ln(v1_i) )
    # ln(v2) - ln(v1) = 2*log_s2 - 2*log_s1 = term_log
    
    elementwise_kl = 0.5 * (term_trace_qc - 1 + term_log)
    return elementwise_kl.sum(dim=-1)

def kl_categorical_logits(u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """
    Computes KL(Cat(softmax(u1)) || Cat(softmax(u2))) stably from logits.
    
    Args:
        u1, u2: Logits.
        Shapes must be broadcastable.
        
    Returns:
        KL divergence summed over the last dim (K).
    """
    lp1 = F.log_softmax(u1, dim=-1)
    lp2 = F.log_softmax(u2, dim=-1)
    p1 = torch.exp(lp1)
    
    # KL = sum p1 * (log p1 - log p2)
    #    = sum p1 * (lp1 - lp2)
    return (p1 * (lp1 - lp2)).sum(dim=-1)
