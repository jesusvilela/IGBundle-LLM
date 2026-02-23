
import torch
import torch.nn as nn
from typing import Optional

class FiberLatentStore(nn.Module):
    """
    Manages the persistent latent vectors (s_i) for each fiber in the bundle.
    Supports snapshotting for retrospective refinement and drift calculation.
    """
    def __init__(self, n_fibers: int, d_s: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.n_fibers = n_fibers
        self.d_s = d_s
        
        # The learnable/evolvable latent state of each fiber
        self.s = nn.Parameter(torch.zeros(n_fibers, d_s, device=device, dtype=dtype))
        
        # Phase 2: Momentum State (p) for Hamiltonian Dynamics
        # Represented as tangent vectors at the origin (or transported to s)
        self.p = nn.Parameter(torch.zeros(n_fibers, d_s, device=device, dtype=dtype))
        
        # Snapshots for refinement and drift calculation
        self.s_prev: Optional[torch.Tensor] = None
        self.s_prev2: Optional[torch.Tensor] = None
        self.p_prev: Optional[torch.Tensor] = None
        
        # Initialize with small random noise
        nn.init.normal_(self.s, std=0.01)
        nn.init.normal_(self.p, std=0.01) # Thermal initialization

    def snapshot(self):
        """Take a snapshot of the current state."""
        self.s_prev2 = None if self.s_prev is None else self.s_prev.clone().detach()
        self.s_prev = self.s.clone().detach()
        self.p_prev = self.p.clone().detach()

    def get(self, fiber_idx: torch.Tensor) -> torch.Tensor:
        """
        Retrieve latents for specific fibers.
        Args:
            fiber_idx: (B, k) or (k,) indices
        Returns:
            (B, k, d_s) or (k, d_s) latents
        """
        return F.embedding(fiber_idx, self.s)

    def forward(self, fiber_idx: torch.Tensor) -> torch.Tensor:
        return self.get(fiber_idx)
        
    def extra_repr(self):
        return f"n_fibers={self.n_fibers}, d_s={self.d_s}"
