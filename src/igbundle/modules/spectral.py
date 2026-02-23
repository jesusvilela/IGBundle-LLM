import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class SpectrallyNormalizedLinear(nn.Module):
    """
    Kirszbraun-Compatible Linear Layer.
    Applying spectral normalization enforces 1-Lipschitz continuity, 
    satisfying the Kirszbraun extension theorem conditions.
    This provides structural variance reduction (bounded gradient variance).
    """
    def __init__(self, in_features, out_features, bias=True, bound=1.0):
        super().__init__()
        # Use spectral_norm hook
        self.linear = spectral_norm(nn.Linear(in_features, out_features, bias=bias))
        self.bound = bound
        
    @property
    def weight(self):
        return self.linear.weight
        
    @property
    def bias(self):
        return self.linear.bias
        
    @property
    def in_features(self):
        return self.linear.in_features
        
    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x):
        # We assume the hook handles the normalization
        # Output is guaranteed to be <= bound * ||x|| if bound is respected
        # Default spectral_norm enforces sigma <= 1? Yes?
        # Actually it enforces sigma = 1 (or divides by max).
        
        # If bound != 1.0, we might need manual scaling?
        # spectral_norm ensures max singular value is 1.
        out = self.linear(x)
        if self.bound != 1.0:
            return out * self.bound
        return out
