import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class MixtureState:
    """
    Represents the distributional state of the LLM in the IGBundle scaffold.
    A state is a mixture of P components, each consisting of:
    - A base Gaussian N(m, diag(sigma^2))
    - A fiber Categorical p = softmax(u)
    
    Shapes:
        w_logits: (B, T, P)
        m: (B, T, P, D_lat)
        log_sigma: (B, T, P, D_lat)
        u: (B, T, P, K)
    """
    w_logits: torch.Tensor
    m: torch.Tensor
    log_sigma: torch.Tensor
    u: torch.Tensor

    def __post_init__(self):
        # Allow init with None for placeholder logic if needed, but typically should be tensors
        if self.w_logits is not None:
            self.assert_shapes()

    @property
    def w(self) -> torch.Tensor:
        """Mixture weights: (B, T, P)"""
        return F.softmax(self.w_logits, dim=-1)

    @property
    def sigma(self) -> torch.Tensor:
        """Standard deviations: (B, T, P, D_lat)"""
        # Clamp for numerical stability
        return torch.exp(self.log_sigma).clamp(min=1e-4, max=1e2)
    
    @property
    def variance(self) -> torch.Tensor:
        """Variances: (B, T, P, D_lat)"""
        return self.sigma.pow(2)

    @property
    def precision(self) -> torch.Tensor:
        """Precisions (1/variance): (B, T, P, D_lat)"""
        return 1.0 / self.variance

    @property
    def p(self) -> torch.Tensor:
        """Fiber probabilities: (B, T, P, K)"""
        return F.softmax(self.u, dim=-1)

    def to(self, device):
        return MixtureState(
            w_logits=self.w_logits.to(device),
            m=self.m.to(device),
            log_sigma=self.log_sigma.to(device),
            u=self.u.to(device)
        )

    def assert_shapes(self):
        B, T, P = self.w_logits.shape
        assert self.m.shape[:3] == (B, T, P), f"m shape mismatch: {self.m.shape} vs {(B, T, P)}"
        assert self.log_sigma.shape == self.m.shape, f"log_sigma shape mismatch: {self.log_sigma.shape}"
        assert self.u.shape[:3] == (B, T, P), f"u shape mismatch: {self.u.shape} vs {(B, T, P)}"
        
    def detach(self):
        return MixtureState(
            w_logits=self.w_logits.detach(),
            m=self.m.detach(),
            log_sigma=self.log_sigma.detach(),
            u=self.u.detach()
        )
