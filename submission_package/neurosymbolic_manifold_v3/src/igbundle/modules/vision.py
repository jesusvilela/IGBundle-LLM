import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class VisionProjector(nn.Module):
    """
    Projector that maps Visual Features (from CLIP/SigLIP) to the Manifold Bottleneck.
    
    Architecture:
    - Input: (B, N, VisionDim)
    - Linear Projection -> Hidden
    - Gated Residual Block (allows flow modulation)
    - Geodesic Alignment Layer (Optional: pre-rotation)
    - Output: (B, N, BottleneckDim)
    """
    def __init__(self, vision_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.vision_dim = vision_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 1. Initial Projection
        self.input_proj = nn.Linear(vision_dim, bottleneck_dim * 2)
        self.norm1 = nn.LayerNorm(bottleneck_dim * 2)
        self.activation = nn.GELU()
        
        # 2. Gated Residual Integration
        # Transforms visual concepts to align with "Text" manifold space
        self.gate_proj = nn.Linear(bottleneck_dim * 2, bottleneck_dim) # Determine "relevance"
        self.feat_proj = nn.Linear(bottleneck_dim * 2, bottleneck_dim) # Transform content
        
        self.norm2 = nn.LayerNorm(bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Geometric Bias (Optional)
        # Learnable "Center" for visual concept cluster
        self.visual_bias = nn.Parameter(torch.zeros(1, 1, bottleneck_dim))

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N, V_dim)
        Returns:
            projected: (B, N, D_bot)
        """
        B, N, _ = vision_features.shape
        
        # 1. Project to intermediate space
        x = self.input_proj(vision_features) # (B, N, D*2)
        x = self.norm1(x)
        x = self.activation(x)
        
        # 2. Gated Projection
        gate = torch.sigmoid(self.gate_proj(x)) # (B, N, D)
        feat = self.feat_proj(x)               # (B, N, D)
        
        out = gate * feat # Gated feature flow
        out = self.norm2(out)
        out = self.dropout(out)
        
        # 3. Add Bias
        out = out + self.visual_bias
        
        return out
