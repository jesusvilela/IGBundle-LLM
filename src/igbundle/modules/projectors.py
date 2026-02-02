import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ProjectorBase(nn.Module, ABC):
    """
    Abstract Base Class for Multimodal Projectors.
    Maps modality-specific embeddings (Vision, Audio) into the Language Model's embedding space.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [Batch, Seq, InputDim]
        Returns:
            Projected features [Batch, Seq, OutputDim]
        """
        pass

class LinearProjector(ProjectorBase):
    """Simple Linear Projection"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class MLPProjector(ProjectorBase):
    """MLP Projection with GELU (LLaVA style)"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GeodesicVisionProjector(ProjectorBase):
    """
    Projects Vision features onto the Manifold's Tangent Space.
    (Epic 8 Specific)
    """
    def __init__(self, input_dim: int, output_dim: int, manifold_dim: int = 32):
        super().__init__(input_dim, output_dim)
        # 1. Compress Vision Features to Manifold Dim
        self.compress = nn.Linear(input_dim, manifold_dim)
        
        # 2. "Exp" map approximation (Learnable)
        # Maps compressed features to Geodesic coordinates relative to a learnable origin
        self.exp_map = nn.Sequential(
            nn.Linear(manifold_dim, manifold_dim),
            nn.Tanh() # Bounded for Poincare/Hyperbolic stability
        )
        
        # 3. Project up to LLM Dim
        self.uplift = nn.Linear(manifold_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, VisionDim]
        
        # Compress
        z = self.compress(x) # [B, S, ManifoldDim]
        
        # Apply Logic/Geometric transformation
        geo_z = self.exp_map(z) 
        
        # Uplift to LLM space
        out = self.uplift(geo_z) # [B, S, LLMDim]
        
        return out
