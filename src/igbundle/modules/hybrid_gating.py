"""
Hybrid Entropy Gating (Epic 37)
-------------------------------
Dynamically weights the contribution of Euclidean vs Riemannian attention
based on the Manifold Entropy (S) of the current state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyGating(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Epic 37 Fix: Gating operates on the Latent Manifold state, not the full residual stream.
        self.dim = getattr(config, 'latent_dim', config.hidden_size)
        
        # Learnable gating network
        # Input: [Entropy_Scalar, Hidden_State_Mean, Curvature_Scalar]
        # Output: lambda (0 to 1) mixing factor
        # lambda = 1 -> Pure Euclidean
        # lambda = 0 -> Pure Riemannian
        
        # Or simpler: just a learned sigmoid over entropy?
        # S high -> Uncertainty -> Use Riemannian (Logic) -> lambda low?
        # S low -> Certainty -> Use Euclidean (Rote) -> lambda high?
        
        # Let's make it state-dependent.
        self.gate_net = nn.Sequential(
            nn.Linear(self.dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                entropy: torch.Tensor, 
                curvature: torch.Tensor) -> torch.Tensor:
        """
        Compute mixing factor lambda.
        
        Args:
            hidden_states: (B, T, D)
            entropy: (B, T) or (B,)
            curvature: (B, T) or (B,)
            
        Returns:
            lambda_weight: (B, T, 1)
        """
        # Feature engineering
        # Append S and K to hidden state
        if entropy.dim() == 1:
             entropy = entropy.unsqueeze(-1).expand(hidden_states.shape[0], hidden_states.shape[1])
        if curvature.dim() == 1:
             curvature = curvature.unsqueeze(-1).expand(hidden_states.shape[0], hidden_states.shape[1])
             
        # Ensure shapes match (B, T, 1)
        s_feat = entropy.unsqueeze(-1)
        k_feat = curvature.unsqueeze(-1)
        
        features = torch.cat([hidden_states, s_feat, k_feat], dim=-1)
        
        lambda_val = self.gate_net(features)
        return lambda_val

class HybridAttentionHead(nn.Module):
    """
    Composes Euclidean and Riemannian Attention.
    """
    def __init__(self, config, euclidean_attn_module, riemannian_attn_module):
        super().__init__()
        self.euclidean = euclidean_attn_module
        self.riemannian = riemannian_attn_module
        self.gating = EntropyGating(config)
        
    def forward(self, hidden_states, key_value_states, 
                entropy, curvature, 
                *args, **kwargs):
                
        # 1. Compute Heads
        out_E = self.euclidean(hidden_states, key_value_states, *args, **kwargs)
        out_R = self.riemannian(hidden_states, key_value_states, *args, **kwargs) # Wraps logic
        
        # 2. Compute Gate
        # lambda * E + (1-lambda) * R
        lam = self.gating(hidden_states, entropy, curvature)
        
        return lam * out_E + (1.0 - lam) * out_R
