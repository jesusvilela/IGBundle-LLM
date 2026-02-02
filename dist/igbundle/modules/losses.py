import torch
import torch.nn.functional as F
from .kl import kl_categorical_logits

def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    where m = 0.5 * (p + q)
    p, q are probabilities (not logits).
    """
    m = 0.5 * (p + q)
    # KL(p||m) = sum p * log(p/m)
    # Add epsilon to prevent log(0)
    kl_pm = (p * (torch.log(p + epsilon) - torch.log(m + epsilon))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + epsilon) - torch.log(m + epsilon))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

class SheafLoss(torch.nn.Module):
    def __init__(self, num_patches, d_lat, tau=1.0):
        super().__init__()
        self.patch_centers = torch.nn.Parameter(torch.randn(num_patches, d_lat))
        self.tau = tau
        
    def forward(self, mixture_state):
        """
        Computes sheaf gluing loss.
        
        Args:
            mixture_state: MixtureState object
            
        Returns:
            loss: scalar
        """
        # mixture_state.m: (B, T, P, D)
        # We need to compute responsibilities of each component i to patch r
        # Dist: (B, T, P, R)
        
        # Cast to float32 for stability and einsum compatibility if mixed precision
        m = mixture_state.m.float() # (B, T, P, D)
        w = mixture_state.w.float() # (B, T, P)
        p = mixture_state.p.float() # (B, T, P, K)
        
        # Ensure centers are float32 (local cast, do not mutate parameter .data in forward)
        patch_centers_f = self.patch_centers.float()
        
        B, T, P, D = m.shape
        R = patch_centers_f.shape[0]
        K = p.shape[-1]
        
        # Expand for broadcasting
        # m: (B, T, P, 1, D)
        # centers: (1, 1, 1, R, D)
        m_exp = m.unsqueeze(-2)
        c_exp = patch_centers_f.reshape(1, 1, 1, R, D)
        
        dists = ((m_exp - c_exp).pow(2).sum(dim=-1)) # (B, T, P, R)
        gamma = F.softmax(-dists / self.tau, dim=-1) # (B, T, P, R) resp of patch r for comp i
        
        # Patch-wise fiber belief: p_bar_r = sum_i gamma_ir * w_i * p_i
        # We need to account for w_i, total mass contribution.
        w_exp = w.unsqueeze(-1) # (B, T, P, 1)
        start_mass = gamma * w_exp # (B, T, P, R)
        total_mass_r = start_mass.sum(dim=2) # (B, T, R)
        
        # Weighted sum of p_i: (B, T, P, K) * (B, T, P, R) -> need careful matmul
        p_bar_num = torch.einsum('btpk,btpr->btrk', p, start_mass)
        p_bar = p_bar_num / (total_mass_r.unsqueeze(-1) + 1e-6) # (B, T, R, K)
        
        # Pairwise JS between patches weighted by patch overlap
        c_i = patch_centers_f.unsqueeze(1)
        c_j = patch_centers_f.unsqueeze(0)
        center_dists = (c_i - c_j).pow(2).sum(dim=-1)
        omega = torch.exp(-center_dists / self.tau) # (R, R)
        
        # Vectorized JS
        # p_r: (B, T, R, 1, K)
        # p_s: (B, T, 1, R, K)
        p_r = p_bar.unsqueeze(3)
        p_s = p_bar.unsqueeze(2)
        
        m_dist = 0.5 * (p_r + p_s)
        # Refactor (Review 2.E): Use unified JS divergence helper
        js_rs = jensen_shannon_divergence(p_r, p_s, epsilon=1e-8) # (B, T, R, R)
        
        weighted_js = omega * js_rs
        
        # Mask diagonal and lower triangle to count each pair once
        mask = torch.triu(torch.ones(R, R, device=p.device), diagonal=1)
        
        # Normalize by (B * T * num_pairs) to make loss scale invariant to batch/seq_len
        pair_count = mask.sum().float()
        total_elements = (B * T * pair_count) + 1e-8
        
        loss = (weighted_js * mask).sum().float() / total_elements
        
        return loss

