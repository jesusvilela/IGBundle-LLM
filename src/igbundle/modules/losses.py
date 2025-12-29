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
        
        # Ensure centers are float32
        self.patch_centers.data = self.patch_centers.data.float()
        
        B, T, P, D = m.shape
        R = self.patch_centers.shape[0]
        K = p.shape[-1]
        
        # Expand for broadcasting
        # m: (B, T, P, 1, D)
        # centers: (1, 1, 1, R, D)
        m_exp = m.unsqueeze(-2)
        c_exp = self.patch_centers.reshape(1, 1, 1, R, D)
        
        dists = ((m_exp - c_exp).pow(2).sum(dim=-1)) # (B, T, P, R)
        gamma = F.softmax(-dists / self.tau, dim=-1) # (B, T, P, R) resp of patch r for comp i
        
        # Patch-wise fiber belief: p_bar_r = sum_i gamma_ir * w_i * p_i
        # We need to account for w_i.
        # w is (B, T, P). Should we weight by w_i? Yes, total mass contribution.
        # But we need p_bar_r to be a probability distribution.
        # So we normalize by sum_i gamma_ir * w_i
        
        w_exp = w.unsqueeze(-1) # (B, T, P, 1)
        start_mass = gamma * w_exp # (B, T, P, R)
        total_mass_r = start_mass.sum(dim=2) # (B, T, R)
        
        # Weighted sum of p_i: (B, T, P, K) * (B, T, P, R) -> need careful matmul
        # p: (B, T, P, K)
        # start_mass: (B, T, P, R)
        # p_bar_num: (B, T, R, K) = sum_p start_mass[...,r] * p[...,k]
        # p[b,t,i,k] * start_mass[b,t,i,r]
        
        p_bar_num = torch.einsum('btpk,btpr->btrk', p, start_mass)
        p_bar = p_bar_num / (total_mass_r.unsqueeze(-1) + 1e-6) # (B, T, R, K)
        
        # Pairwise JS between patches?
        # Only for overlapping patches?
        # Simplified: all pairs or nearest neighbors.
        # Let's do all pairs weighted by patch overlap (implicit in geometry) 
        # or just sum over r<s JS(p_bar_r, p_bar_s) if we assume all should be consistent?
        # User script: "sum_{r<s} omega_rs * JS(p_bar_r, p_bar_s)"
        # omega_rs depends on patch overlap.
        
        # Compute patch distances
        c_i = self.patch_centers.unsqueeze(1)
        c_j = self.patch_centers.unsqueeze(0)
        center_dists = (c_i - c_j).pow(2).sum(dim=-1)
        omega = torch.exp(-center_dists / self.tau) # (R, R)
        
        # JS calculation
        # p_bar: (B, T, R, K)
        
        # We can implement a simplified version summing over random pairs or all pairs
        # For efficiency, let's just do a mean JS of all pairs weighted by omega
        
        # This is expensive O(R^2). R is small (8).
        loss = 0.0
        # Iterate or vectorized? R=8 is small enough to loop or vectorize
        
        # Vectorized JS
        # p_r: (B, T, R, 1, K)
        # p_s: (B, T, 1, R, K)
        # Broadcst to (B, T, R, R, K)
        p_r = p_bar.unsqueeze(3)
        p_s = p_bar.unsqueeze(2)
        
        m_dist = 0.5 * (p_r + p_s)
        # KL(p_r || m)
        kl_r = (p_r * (torch.log(p_r + 1e-8) - torch.log(m_dist + 1e-8))).sum(dim=-1)
        kl_s = (p_s * (torch.log(p_s + 1e-8) - torch.log(m_dist + 1e-8))).sum(dim=-1)
        js_rs = 0.5 * (kl_r + kl_s) # (B, T, R, R)
        
        weighted_js = omega * js_rs
        
        # Mask diagonal and lower triangle to count each pair once
        mask = torch.triu(torch.ones(R, R, device=p.device), diagonal=1)
        # Sum with epsilon and float32 accumulation
        loss = (weighted_js * mask).sum().float() / (mask.sum().float() + 1e-8)
        
        return loss

