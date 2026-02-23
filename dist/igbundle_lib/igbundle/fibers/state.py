from dataclasses import dataclass
import torch

@dataclass
class FiberState:
    z: torch.Tensor          # Shape [d], the position q on manifold
    p: torch.Tensor          # Shape [d], the momentum p (Tangent Space)
    chart_idx: int = 0       # For multi-chart atlases
    log_metric: torch.Tensor = None # Parameterized metric G (optional)
    
    def to(self, device):
        self.z = self.z.to(device)
        self.p = self.p.to(device)
        if self.log_metric is not None:
             self.log_metric = self.log_metric.to(device)
        return self

    @staticmethod
    def initialize(dim: int, device: str = "cuda") -> 'FiberState':
        # Initialize q near origin
        q = torch.randn(dim, device=device) * 0.01
        # Initialize p from Gaussian (Thermal eq)
        p = torch.randn(dim, device=device) * 0.1
        return FiberState(z=q, p=p)
