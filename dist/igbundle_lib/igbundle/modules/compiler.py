import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldDecompiler(nn.Module):
    """
    Translates Continuous Manifold States into Discrete Logical Operations.
    """
    def __init__(self, latent_dim: int, num_ops: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_ops = num_ops
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, num_ops)
        )
        
    def forward(self, manifold_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            manifold_coords: (B, T, P, D)
        Returns:
            op_logits: (B, T, NumOps)
        """
        mean_coords = manifold_coords.mean(dim=2) # (B, T, D)
        logits = self.decoder(mean_coords)
        return logits
    
    def decode_sequence(self, manifold_coords: torch.Tensor) -> torch.Tensor:
        logits = self.forward(manifold_coords)
        return torch.argmax(logits, dim=-1)
