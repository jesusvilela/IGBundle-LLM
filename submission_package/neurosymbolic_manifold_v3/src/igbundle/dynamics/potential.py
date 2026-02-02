import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralPotential(nn.Module):
    """
    Learned Potential Energy V(q) for the Hamiltonian System.
    
    The potential acts as the "Semantic Landscape" or "Gravity".
    High Potential = Implausible/Incorrect States (Repulses particles)
    Low Potential = Plausible/Correct States (Attracts particles)
    
    H(q, p) = T(p) + V(q)
    
    Dynamics:
    dq/dt = dT/dp (Velocity)
    dp/dt = -dV/dq (Force)
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        
        layers = []
        # Input: q of shape [..., latent_dim]
        # We process each point independently (or could use attention if q describes a set)
        
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.SiLU()) # Smooth activation for smooth potential surface
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            
        # Output: Scalar Potential Energy
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize to near-zero to start with "Free Particle" dynamics
        # This ensures we don't disrupt the pretrained "Kinetic" phase initially.
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute Potential Energy V(q).
        Args:
            q: Latent coordinates [Batch, ..., LatentDim]
        Returns:
            v: Scalar potential [Batch, ..., 1]
        """
        return self.net(q)
