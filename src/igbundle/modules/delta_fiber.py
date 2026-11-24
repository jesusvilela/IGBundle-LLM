"""Opt-in gated delta-rule recurrence for fiber sections.

Ported from development commit db00915. The state is an associative memory
matrix. Optional curvature and entropy inputs are scalar gating features.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class DeltaFiberUpdate(nn.Module):
    """Process (batch, tokens, components, coordinates + categories).

    Memory starts at zero on each call and is independent across batches,
    components and heads. Each token erases and writes before it reads:

        M_t = decay * M - alpha * (M k) k^T + beta * v k^T
        output_t = out_proj(concat_heads(M_t q))
    The default decay is one. Passing erase=False disables the erase term.
    """

    def __init__(
        self, coord_dim: int, section_dim: int, mem_dim: int = 64,
        num_heads: int = 4, *, erase: bool = True, decay: float = 1.0,
    ):
        super().__init__()
        if min(coord_dim, section_dim, mem_dim, num_heads) <= 0 or mem_dim % num_heads:
            raise ValueError("Dimensions must be positive and mem_dim divisible by num_heads")
        if mem_dim // num_heads < 2:
            raise ValueError("Layer-normalized queries and keys require at least two coordinates per head")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("Memory decay must be between zero and one")
        self.erase = erase
        self.decay = decay
        self.D = coord_dim
        self.K = section_dim
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        self.head_dim = mem_dim // num_heads

        joint_dim = coord_dim + section_dim
        self.W_q = nn.Linear(joint_dim, mem_dim, bias=False)
        self.W_k = nn.Linear(joint_dim, mem_dim, bias=False)
        self.W_v = nn.Linear(joint_dim, mem_dim, bias=False)
        self.W_beta = nn.Linear(joint_dim, num_heads)
        self.W_alpha = nn.Linear(joint_dim, num_heads)
        self.curvature_mod = nn.Linear(1, num_heads, bias=False)
        self.entropy_mod = nn.Linear(1, num_heads, bias=False)
        self.out_proj = nn.Linear(mem_dim, section_dim)
        self.norm_q = nn.LayerNorm(self.head_dim)
        self.norm_k = nn.LayerNorm(self.head_dim)

        for projection in (self.W_q, self.W_k, self.W_v):
            nn.init.normal_(projection.weight, std=0.02)
        nn.init.constant_(self.W_beta.bias, 0.5)
        nn.init.constant_(self.W_alpha.bias, -1.0)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.curvature_mod.weight, std=0.001)
        nn.init.normal_(self.entropy_mod.weight, std=0.001)

    def forward(
        self,
        joint_repr: torch.Tensor,
        curvature: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (batch, tokens, components, categories) additive updates.

        Curvature and entropy accept scalar or per-batch tensors. Their
        values are detached, while the modulation weights remain trainable.
        """
        if joint_repr.ndim != 4 or joint_repr.shape[-1] != self.D + self.K:
            raise ValueError("Expected (batch, tokens, components, coordinates + categories)")
        batch, tokens, components, features = joint_repr.shape
        heads, dim = self.num_heads, self.head_dim
        x = joint_repr.transpose(1, 2).reshape(batch * components, tokens, features)
        q = self.norm_q(self.W_q(x).unflatten(-1, (heads, dim)))
        k = self.norm_k(self.W_k(x).unflatten(-1, (heads, dim)))
        k = F.normalize(k, dim=-1, eps=max(1e-12, torch.finfo(k.dtype).tiny))
        v = self.W_v(x).unflatten(-1, (heads, dim))

        beta = self.W_beta(x)
        alpha = self.W_alpha(x)
        if curvature is not None:
            feature = self._batch_feature(curvature.detach().abs().clamp(max=10), batch, components, x)
            beta = beta + self.curvature_mod(feature)
        if entropy is not None:
            feature = self._batch_feature(entropy.detach().clamp(0, 5), batch, components, x)
            alpha = alpha + self.entropy_mod(feature)
        beta = beta.sigmoid()[..., None, None]
        alpha = alpha.sigmoid()[..., None, None]

        memory = v.new_zeros(batch * components, heads, dim, dim)
        reads = []
        for t in range(tokens):
            key = k[:, t, :, :, None]
            value = v[:, t, :, :, None]
            query = q[:, t, :, :, None]
            key_transpose = key.transpose(-2, -1)
            correction = beta[:, t] * value
            if self.erase:
                correction = correction - alpha[:, t] * (memory @ key)
            memory = self.decay * memory + correction @ key_transpose
            reads.append((memory @ query).squeeze(-1))

        readout = torch.stack(reads, dim=1).flatten(-2) if reads else v.flatten(-2)
        update = self.out_proj(readout)
        return update.reshape(batch, components, tokens, self.K).transpose(1, 2)

    @staticmethod
    def _batch_feature(values, batch, components, reference):
        if values.ndim > 1 or values.numel() not in (1, batch):
            raise ValueError("Expected a scalar or one gating value per batch item")
        return values.to(reference).reshape(-1).expand(batch).repeat_interleave(components)[:, None, None]
