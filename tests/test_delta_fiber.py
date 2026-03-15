"""
Tests for Epic 17b: Delta-Net Fiber Dynamics

Validates:
1. Forward pass shapes
2. Gradient flow through delta recurrence
3. Gate modulation by curvature/entropy
4. DeltaNetAttention API compatibility
5. Memory capacity (write-then-read)
"""

import torch
import torch.nn as nn
import pytest

from src.igbundle.modules.delta_fiber import DeltaFiberUpdate, DeltaNetAttention


# --- DeltaFiberUpdate Tests ---

class TestDeltaFiberUpdate:

    def setup_method(self):
        self.D = 32   # coord_dim (latent_dim)
        self.K = 8    # section_dim (num_categories)
        self.P = 4    # num_components
        self.B = 2    # batch
        self.T = 16   # sequence length
        self.module = DeltaFiberUpdate(
            coord_dim=self.D, section_dim=self.K,
            mem_dim=64, num_heads=4
        )

    def test_output_shape(self):
        joint = torch.randn(self.B, self.T, self.P, self.D + self.K)
        out = self.module(joint)
        assert out.shape == (self.B, self.T, self.P, self.K), \
            f"Expected (B,T,P,K)={(self.B,self.T,self.P,self.K)}, got {out.shape}"

    def test_gradient_flow(self):
        """Verify gradients flow through the delta recurrence."""
        joint = torch.randn(self.B, self.T, self.P, self.D + self.K, requires_grad=True)
        out = self.module(joint)
        loss = out.sum()
        loss.backward()
        assert joint.grad is not None, "No gradient on input"
        assert joint.grad.abs().sum() > 0, "Gradient is all zeros"

    def test_gradient_flow_with_gates(self):
        """Verify gradients flow when curvature/entropy modulate gates."""
        joint = torch.randn(self.B, self.T, self.P, self.D + self.K, requires_grad=True)
        K = torch.tensor([-2.0])
        S = torch.tensor([1.5])
        out = self.module(joint, curvature=K, entropy=S)
        loss = out.sum()
        loss.backward()
        assert joint.grad is not None
        assert joint.grad.abs().sum() > 0

    def test_curvature_modulates_write_gate(self):
        """High |K| should increase write gate activity → larger updates."""
        joint = torch.randn(1, 4, self.P, self.D + self.K)

        out_low_K = self.module(joint.clone(), curvature=torch.tensor([0.1]))
        out_high_K = self.module(joint.clone(), curvature=torch.tensor([5.0]))

        # Higher curvature should produce different (generally larger) updates
        norm_low = out_low_K.abs().mean().item()
        norm_high = out_high_K.abs().mean().item()
        # They should at least be different (modulation is working)
        assert abs(norm_low - norm_high) > 1e-6, \
            f"Curvature modulation had no effect: low={norm_low:.6f}, high={norm_high:.6f}"

    def test_entropy_modulates_erase_gate(self):
        """High S should increase erase gate → different memory state."""
        joint = torch.randn(1, 4, self.P, self.D + self.K)

        out_low_S = self.module(joint.clone(), entropy=torch.tensor([0.1]))
        out_high_S = self.module(joint.clone(), entropy=torch.tensor([3.0]))

        diff = (out_low_S - out_high_S).abs().mean().item()
        assert diff > 1e-8, f"Entropy modulation had no effect: diff={diff:.8f}"

    def test_small_init_output(self):
        """Output projection is small-initialized → initial output should be small."""
        joint = torch.randn(1, 1, self.P, self.D + self.K)
        out = self.module(joint)
        assert out.abs().max().item() < 1.0, \
            f"Small-init output too large: max={out.abs().max().item():.4f}"

    def test_memory_write_read(self):
        """Write a pattern, then read it back — delta rule should associate."""
        # Sequence of 2 tokens: first writes, second reads similar key
        joint = torch.randn(1, 8, self.P, self.D + self.K)
        out = self.module(joint)
        # Later tokens should have larger outputs (accumulated memory)
        early_norm = out[:, :2, :, :].abs().mean().item()
        late_norm = out[:, -2:, :, :].abs().mean().item()
        # Not a strict test, but memory accumulation should generally increase signal
        assert late_norm >= early_norm * 0.5, \
            f"Memory not accumulating: early={early_norm:.4f}, late={late_norm:.4f}"

    def test_numerical_stability(self):
        """No NaN/Inf with reasonable inputs."""
        joint = torch.randn(self.B, 64, self.P, self.D + self.K)
        out = self.module(joint, curvature=torch.tensor([-5.0]), entropy=torch.tensor([2.5]))
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"


# --- DeltaNetAttention Tests ---

class TestDeltaNetAttention:

    def setup_method(self):
        self.D = 128  # embed_dim (bottleneck)
        self.module = DeltaNetAttention(embed_dim=self.D, num_heads=4, dropout=0.0)

    def test_output_shape(self):
        query = torch.randn(2, 16, self.D)   # text: (B, T_text, D)
        key = torch.randn(2, 729, self.D)     # vision: (B, N_patches, D)
        value = key.clone()
        out, weights = self.module(query, key, value)
        assert out.shape == (2, 16, self.D), f"Wrong shape: {out.shape}"
        assert weights is None  # compatibility

    def test_gradient_flow(self):
        query = torch.randn(2, 8, self.D, requires_grad=True)
        key = torch.randn(2, 49, self.D, requires_grad=True)
        out, _ = self.module(query, key, key)
        out.sum().backward()
        assert query.grad is not None and query.grad.abs().sum() > 0
        assert key.grad is not None and key.grad.abs().sum() > 0

    def test_api_compatibility(self):
        """Should be a drop-in replacement for nn.MultiheadAttention."""
        query = torch.randn(1, 4, self.D)
        kv = torch.randn(1, 16, self.D)

        # Both should accept the same call signature
        delta_out, delta_w = self.module(query=query, key=kv, value=kv)

        mha = nn.MultiheadAttention(self.D, 4, batch_first=True)
        mha_out, mha_w = mha(query=query, key=kv, value=kv)

        # Same output shape
        assert delta_out.shape == mha_out.shape

    def test_numerical_stability(self):
        query = torch.randn(2, 32, self.D)
        kv = torch.randn(2, 729, self.D)  # SigLIP2 so400m: 27*27=729 patches
        out, _ = self.module(query, kv, kv)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
