"""
Epic 17b: Delta-Net Fiber Dynamics

Replaces MLP-based fiber_update_net with a gated delta rule —
geometrically motivated linear recurrence on the fiber bundle.

The delta rule is interpreted as parallel transport:
  - Write gate β (curvature-driven): high |K| = high novelty = store new section
  - Erase gate α (entropy-driven): high S = uncertain state = erase stale memory
  - Memory matrix M: persistent fiber state (parallel to FiberLatentStore)

Also provides DeltaNetAttention as O(T) replacement for O(T²) softmax
cross-attention in vision fusion.

Reference: Yang et al. "Parallelizing Linear Transformers with the Delta Rule"
           (ICML 2024, arXiv:2406.06484)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeltaFiberUpdate(nn.Module):
    """
    Gated delta rule update for fiber sections.

    Instead of:
        update = MLP(cat[coords, sections])  # old fiber_update_net

    We compute:
        k_t = W_k · joint_repr           # key: what to index in memory
        v_t = W_v · joint_repr           # value: what to write
        β_t = σ(W_β · joint_repr + b_β)  # write gate (curvature-modulated)
        α_t = σ(W_α · joint_repr + b_α)  # erase gate (entropy-modulated)

        # Delta rule update (associative memory with erase-then-write):
        M_t = M_{t-1} - α_t · (M_{t-1} @ k_t) ⊗ k_t + β_t · v_t ⊗ k_t

        # Read: query the memory for fiber update
        q_t = W_q · joint_repr
        output = M_t @ q_t               # (D_mem,) -> fiber update

    The memory M persists across tokens within a sequence, implementing
    a linear recurrent state that replaces the stateless MLP.

    Geometric interpretation:
        - M is a connection form on the fiber bundle
        - The delta update performs parallel transport along the base manifold
        - β gates new curvature information into the connection
        - α forgets stale connection coefficients when entropy is high
    """

    def __init__(
        self,
        coord_dim: int,       # D = latent_dim (32)
        section_dim: int,     # K = num_categories (8)
        mem_dim: int = 64,    # memory key/value dimension
        num_heads: int = 4,   # multi-head delta for capacity
    ):
        super().__init__()
        self.D = coord_dim
        self.K = section_dim
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        self.head_dim = mem_dim // num_heads

        joint_dim = coord_dim + section_dim  # input: cat[coords, sections]

        # Projections
        self.W_q = nn.Linear(joint_dim, mem_dim, bias=False)
        self.W_k = nn.Linear(joint_dim, mem_dim, bias=False)
        self.W_v = nn.Linear(joint_dim, mem_dim, bias=False)

        # Gates — initialized to moderate values for stability
        self.W_beta = nn.Linear(joint_dim, num_heads)   # write gate per head
        self.W_alpha = nn.Linear(joint_dim, num_heads)  # erase gate per head

        # Curvature and entropy modulation projections
        # These allow K and S to directly influence the gates
        self.curvature_mod = nn.Linear(1, num_heads, bias=False)
        self.entropy_mod = nn.Linear(1, num_heads, bias=False)

        # Output: map memory readout -> fiber section update
        self.out_proj = nn.Linear(mem_dim, section_dim)

        # Layer norm for stability
        self.norm_q = nn.LayerNorm(self.head_dim)
        self.norm_k = nn.LayerNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        # Small init for stability — delta rule can amplify
        for p in [self.W_q, self.W_k, self.W_v]:
            nn.init.normal_(p.weight, std=0.02)
        # Gates biased toward moderate write, low erase at start
        nn.init.constant_(self.W_beta.bias, 0.5)   # β ≈ 0.62 at init
        nn.init.constant_(self.W_alpha.bias, -1.0)  # α ≈ 0.27 at init
        # Small non-zero init so gradients flow from the start
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)
        # Curvature/entropy modulation starts near-zero (additive)
        nn.init.normal_(self.curvature_mod.weight, std=0.001)
        nn.init.normal_(self.entropy_mod.weight, std=0.001)

    def forward(
        self,
        joint_repr: torch.Tensor,       # (B, T, P, D+K)
        curvature: Optional[torch.Tensor] = None,  # (B,) or scalar
        entropy: Optional[torch.Tensor] = None,     # (B,) or scalar
    ) -> torch.Tensor:
        """
        Args:
            joint_repr: concatenation of [coords, sections], shape (B, T, P, D+K)
            curvature: current sectional curvature K (scalar or per-batch)
            entropy: current fiber entropy S (scalar or per-batch)

        Returns:
            fiber_update: (B, T, P, K) — additive update to fiber sections
        """
        B, T, P, J = joint_repr.shape
        H = self.num_heads
        d = self.head_dim

        # Flatten P into batch for parallel processing across fiber components
        x = joint_repr.reshape(B * P, T, J)  # (B*P, T, J)

        # Project to queries, keys, values
        q = self.W_q(x).view(B * P, T, H, d)  # (B*P, T, H, d)
        k = self.W_k(x).view(B * P, T, H, d)
        v = self.W_v(x).view(B * P, T, H, d)

        # Normalize for numerical stability
        q = self.norm_q(q)
        k = self.norm_k(k)
        # L2 normalize keys (critical for delta rule stability)
        k = F.normalize(k, dim=-1)

        # Compute gates
        beta = torch.sigmoid(self.W_beta(x))   # (B*P, T, H) — write gate
        alpha = torch.sigmoid(self.W_alpha(x))  # (B*P, T, H) — erase gate

        # Modulate gates with curvature and entropy
        if curvature is not None:
            K_val = curvature.detach().float()
            if K_val.dim() == 0:
                K_val = K_val.unsqueeze(0)
            # Expand scalar/batch to (B,) then repeat for P components → (B*P,)
            if K_val.shape[0] < B:
                K_val = K_val.expand(B)
            K_val = K_val.abs().clamp(max=10.0)  # |K|
            # (B,) → (B*P, 1, 1) via repeat_interleave
            K_feat = K_val.repeat_interleave(P).unsqueeze(-1).unsqueeze(-1)  # (B*P, 1, 1)
            K_mod = self.curvature_mod(K_feat)  # (B*P, 1, H)
            beta = torch.sigmoid(beta + K_mod)  # re-apply sigmoid after modulation

        if entropy is not None:
            S_val = entropy.detach().float()
            if S_val.dim() == 0:
                S_val = S_val.unsqueeze(0)
            if S_val.shape[0] < B:
                S_val = S_val.expand(B)
            S_val = S_val.clamp(0, 5.0)
            S_feat = S_val.repeat_interleave(P).unsqueeze(-1).unsqueeze(-1)  # (B*P, 1, 1)
            S_mod = self.entropy_mod(S_feat)  # (B*P, 1, H)
            alpha = torch.sigmoid(alpha + S_mod)  # re-apply sigmoid

        # Reshape gates for broadcasting: (B*P, T, H) → (B*P, T, H, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # (B*P, T, H, 1, 1)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)

        # --- Delta rule recurrence ---
        # M: (B*P, H, d, d) — memory matrix per head
        M = torch.zeros(B * P, H, d, d, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, t, :, :]  # (B*P, H, d)
            v_t = v[:, t, :, :]
            q_t = q[:, t, :, :]
            beta_t = beta[:, t, :, :, :]   # (B*P, H, 1, 1)
            alpha_t = alpha[:, t, :, :, :]

            # Retrieve what memory currently associates with this key
            # M @ k_t: (B*P, H, d, d) @ (B*P, H, d, 1) → (B*P, H, d, 1)
            Mk = torch.matmul(M, k_t.unsqueeze(-1))  # (B*P, H, d, 1)

            # Delta rule: erase old association, write new one
            # M_t = M_{t-1} - α * (M·k) ⊗ k + β * v ⊗ k
            erase = alpha_t * torch.matmul(Mk, k_t.unsqueeze(-2))  # (B*P, H, d, d)
            write = beta_t * torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))  # (B*P, H, d, d)
            M = M - erase + write

            # Read: query the memory
            o_t = torch.matmul(M, q_t.unsqueeze(-1)).squeeze(-1)  # (B*P, H, d)
            outputs.append(o_t)

        # Stack: (B*P, T, H, d) → (B*P, T, H*d) → (B*P, T, mem_dim)
        output = torch.stack(outputs, dim=1).reshape(B * P, T, H * d)

        # Project to fiber section space
        fiber_update = self.out_proj(output)  # (B*P, T, K)
        fiber_update = fiber_update.view(B, P, T, self.K).permute(0, 2, 1, 3)  # (B, T, P, K)

        return fiber_update


class DeltaNetAttention(nn.Module):
    """
    O(T) delta-net linear attention for vision-text cross-attention.

    Replaces nn.MultiheadAttention (O(T²)) in the vision fusion path.

    The delta rule attention computes:
        For each text query token t:
            M_t = M_{t-1} + β_t * (v_t ⊗ k_t - α_t * (M·k_t) ⊗ k_t)
            attn_out_t = M_t @ q_t

    Where k_t, v_t come from vision features (cross-attention),
    and q_t comes from text features.

    For cross-attention, we process all vision tokens first to build M,
    then query with text tokens.
    """

    def __init__(
        self,
        embed_dim: int,     # bottleneck dim
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.beta_proj = nn.Linear(embed_dim, num_heads)
        self.alpha_proj = nn.Linear(embed_dim, num_heads)

        self.norm_q = nn.LayerNorm(self.head_dim)
        self.norm_k = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.beta_proj.bias, 0.0)
        nn.init.constant_(self.alpha_proj.bias, -1.0)

    def forward(
        self,
        query: torch.Tensor,   # (B, T_text, D) — text features
        key: torch.Tensor,     # (B, T_vis, D)  — vision features
        value: torch.Tensor,   # (B, T_vis, D)  — vision features
    ) -> Tuple[torch.Tensor, None]:
        """
        Cross-attention via delta rule:
        1. Build memory M by scanning over vision tokens (key/value)
        2. Query M with text tokens

        Returns: (output, None) — None for attn_weights compat
        """
        B, T_q, D = query.shape
        _, T_kv, _ = key.shape
        H = self.num_heads
        d = self.head_dim

        # Project
        q = self.q_proj(query).view(B, T_q, H, d)
        k = self.k_proj(key).view(B, T_kv, H, d)
        v = self.v_proj(value).view(B, T_kv, H, d)

        # Normalize
        q = self.norm_q(q)
        k = self.norm_k(k)
        k = F.normalize(k, dim=-1)

        # Gates for vision tokens (what to write/erase)
        beta = torch.sigmoid(self.beta_proj(key)).unsqueeze(-1).unsqueeze(-1)   # (B, T_kv, H, 1, 1)
        alpha = torch.sigmoid(self.alpha_proj(key)).unsqueeze(-1).unsqueeze(-1)

        # Phase 1: Build memory by scanning over vision tokens
        M = torch.zeros(B, H, d, d, device=query.device, dtype=query.dtype)
        for t in range(T_kv):
            k_t = k[:, t, :, :]  # (B, H, d)
            v_t = v[:, t, :, :]
            Mk = torch.matmul(M, k_t.unsqueeze(-1))  # (B, H, d, 1)
            erase = alpha[:, t] * torch.matmul(Mk, k_t.unsqueeze(-2))
            write = beta[:, t] * torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))
            M = M - erase + write

        # Phase 2: Query memory with text tokens (all in parallel — M is fixed)
        # q: (B, T_q, H, d) → (B, H, T_q, d)
        q_perm = q.permute(0, 2, 1, 3)
        # M @ q^T: (B, H, d, d) @ (B, H, d, T_q) → (B, H, d, T_q)
        output = torch.matmul(M, q_perm.transpose(-1, -2))  # (B, H, d, T_q)
        output = output.permute(0, 3, 1, 2).reshape(B, T_q, D)  # (B, T_q, D)

        output = self.out_proj(output)
        output = self.dropout(output)

        return output, None  # None for compatibility with nn.MultiheadAttention API
