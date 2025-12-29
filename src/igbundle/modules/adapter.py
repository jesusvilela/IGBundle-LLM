import torch
import torch.nn as nn
import torch.nn.functional as F
from .state import MixtureState
from .kl import kl_diag_gauss, kl_categorical_logits
from .ops import bundle_affinity, mix_messages

class IGBundleAdapter(nn.Module):
    def __init__(self, config):
        """
        config: object with attributes:
            hidden_size: int (backbone hidden dim)
            bottleneck_dim: int (optional, defaults to hidden_size // 4)
            num_components: int (P)
            num_categories: int (K)
            latent_dim: int (D_lat, usually hidden_size or smaller)
            alpha: float (Base KL weight)
            beta: float (Fiber KL weight)
            eta_f: float (Fiber learning rate/step size)
            eta_b: float (Base learning rate/step size)
            eta_w: float (Weight learning rate/step size)
            adapter_scale: float
            dropout: float
        """
        super().__init__()
        self.cfg = config
        
        self.P = config.num_components
        self.K = config.num_categories
        self.D_lat = config.latent_dim
        self.H = config.hidden_size
        
        # Bottleneck Dimension
        self.D_bot = getattr(config, 'bottleneck_dim', self.H // 4)
        print(f"IGBundleAdapter: Using bottleneck_dim={self.D_bot} (H={self.H})")
        
        # 1. Input Projection (H -> D_bot)
        self.input_proj = nn.Linear(self.H, self.D_bot)
        
        # 2. Projections to Mixture Params (from D_bot)
        # These were the heavy layers (H -> P*D)
        # Now (D_bot -> P*D)
        self.proj_w = nn.Linear(self.D_bot, self.P)
        self.proj_m = nn.Linear(self.D_bot, self.P * self.D_lat)
        self.proj_ls = nn.Linear(self.D_bot, self.P * self.D_lat)
        self.proj_u = nn.Linear(self.D_bot, self.P * self.K)
        
        # Message processing (phi)
        # Input: m, log_sigma, u, context -> message
        msg_input_dim = self.D_lat * 2 + self.K
        # Inner dim of message processor: let's use D_bot instead of H for efficiency
        self.msg_processor = nn.Sequential(
            nn.Linear(msg_input_dim, self.D_bot), 
            nn.GELU(),
            nn.Linear(self.D_bot, self.D_bot) 
        )
        
        # Heads for IG scores (from D_bot)
        self.head_s_u = nn.Linear(self.D_bot, self.K)
        self.head_g_m = nn.Linear(self.D_bot, self.D_lat)
        self.head_g_ls = nn.Linear(self.D_bot, self.D_lat)
        self.head_r_w = nn.Linear(self.D_bot, 1)
        
        # Output projection
        # self.comp_out: (2*D_lat + K) -> D_bot
        self.comp_out = nn.Linear(self.D_lat * 2 + self.K, self.D_bot)
        
        # Final projection: D_bot -> H
        self.out_proj = nn.Linear(self.D_bot, self.H)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = config.adapter_scale
        
        # Init weights (optional, typically small for adapters)
        nn.init.normal_(self.input_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.weight) # Zero init for identity start

    def forward(self, x, context=None):
        """
        x: (B, T, H)
        context: (B, T, H) optional
        """
        B, T, H = x.shape
        
        # 0. Bottleneck
        h_bot = self.input_proj(x) # (B, T, D_bot)
        
        # 1. Project to Mixture Params using h_bot
        w_logits = self.proj_w(h_bot) # (B, T, P)
        m = self.proj_m(h_bot).view(B, T, self.P, self.D_lat)
        log_sigma = self.proj_ls(h_bot).view(B, T, self.P, self.D_lat)
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=5.0)
        u = self.proj_u(h_bot).view(B, T, self.P, self.K)
        
        state = MixtureState(w_logits, m, log_sigma, u)
        
        # 2. Compute Bundle Affinity (Same as before)
        m_i = m.unsqueeze(3) # (B, T, P, 1, D)
        ls_i = log_sigma.unsqueeze(3)
        u_i = u.unsqueeze(3)
        
        m_j = m.unsqueeze(2) # (B, T, 1, P, D)
        ls_j = log_sigma.unsqueeze(2)
        u_j = u.unsqueeze(2)
        
        d_base = kl_diag_gauss(m_i, ls_i, m_j, ls_j)
        d_fiber = kl_categorical_logits(u_i, u_j)
        A = bundle_affinity(d_base, d_fiber, self.cfg.alpha, self.cfg.beta) # (B, T, P, P)
        
        # 3. Message Passing
        raw_feats = torch.cat([m, log_sigma, u], dim=-1)
        processed_feats = self.msg_processor(raw_feats) # (B, T, P, D_bot)
        
        mixed_msg = mix_messages(A, processed_feats) # (B, T, P, D_bot)
        
        # 4. Compute Scores (from mixed_msg which is D_bot)
        s_u = self.head_s_u(mixed_msg).view(B, T, self.P, self.K)
        g_m = self.head_g_m(mixed_msg).view(B, T, self.P, self.D_lat)
        g_ls = self.head_g_ls(mixed_msg).view(B, T, self.P, self.D_lat)
        r_w = self.head_r_w(mixed_msg).view(B, T, self.P)
        
        # 5. Apply IG Updates (Same as before)
        u_new = u + self.cfg.eta_f * s_u
        
        prec = torch.exp(-2 * log_sigma)
        prec_new = (prec + self.cfg.eta_b * g_ls).clamp(min=1e-6, max=1e4)
        log_sigma_new = -0.5 * torch.log(prec_new)
        
        m_new = m + self.cfg.eta_b * g_m / (1.0 + prec)
        
        w_logits_new = w_logits + self.cfg.eta_w * r_w
        
        state_new = MixtureState(w_logits_new, m_new, log_sigma_new, u_new)
        
        # 6. Re-encode to hidden
        # Flatten updated params
        raw_feats_new = torch.cat([m_new, log_sigma_new, u_new], dim=-1) # (..., D_latent*2 + K)
        
        # Map back to D_bot
        comp_emb = self.comp_out(raw_feats_new) # (B, T, P, D_bot)
        
        # Weighted sum by new weights
        weights = F.softmax(w_logits_new, dim=-1).unsqueeze(-1) # (B, T, P, 1)
        pooled = (comp_emb * weights).sum(dim=2) # (B, T, D_bot)
        
        # Final up-projection
        out = self.out_proj(pooled) # (B, T, H)
        out = self.dropout(out)
        
        return x + self.scale * out, state_new
