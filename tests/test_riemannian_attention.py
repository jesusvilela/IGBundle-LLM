
import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from igbundle.modules.riemannian_attention import RiemannianAttention
from igbundle.geometry.hyperbolic import PoincareBall

class MockConfig:
    hidden_size = 64
    num_attention_heads = 4
    manifold_curvature = 1.0
    geometric_temperature = 1.0
    learn_geometric_tau = False

class TestRiemannianAttention(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.attn = RiemannianAttention(self.config)
        torch.manual_seed(42)

    def test_forward_shape(self):
        B, H, T, S, D = 1, 4, 2, 5, 16
        # Queries and Keys on Manifold (Projected)
        q = torch.randn(B, H, T, D) * 0.1 # Small norm
        k = torch.randn(B, H, S, D) * 0.1
        v = torch.randn(B, H, S, D) # Values (Euclidean)
        
        out = self.attn(q, k, v)
        self.assertEqual(out.shape, (B, H, T, D))

    def test_attention_focus(self):
        """Test if query attends to closest key."""
        B, H, T, S, D = 1, 1, 1, 3, 4
        
        # q is close to k[0], far from k[1], k[2]
        q = torch.zeros(B, H, T, D) # Origin
        k = torch.zeros(B, H, S, D)
        k[0, 0, 0, :] = torch.tensor([0.1, 0, 0, 0]) # Close
        k[0, 0, 1, :] = torch.tensor([0.8, 0, 0, 0]) # Far
        k[0, 0, 2, :] = torch.tensor([0.9, 0, 0, 0]) # Very Far
        
        v = torch.eye(3).view(B, H, S, 3) # Indicator values [1,0,0], [0,1,0], ...
        # Output dim is 3 here just for test
        
        # Attention should focus on Index 0
        # dist(0, 0.1) ~ small
        # dist(0, 0.8) ~ large (near boundary)
        
        # We need to hack V dim to match D or change logic?
        # The class expects D=head_dim. Here D=4.
        # But matmul is (..., S) @ (..., S, D_v).
        # Let's keep V dim = 4.
        v = torch.zeros(B, H, S, 4)
        v[0,0,0,0] = 100.0 # Target
        
        out = self.attn(q, k, v)
        
        # Should be predominantly 100.0 in first component
        print(f"Attention Out: {out[0,0,0]}")
        self.assertGreater(out[0,0,0,0], 50.0)

if __name__ == '__main__':
    unittest.main()
