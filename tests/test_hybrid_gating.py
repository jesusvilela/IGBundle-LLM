
import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from igbundle.modules.hybrid_gating import EntropyGating

class MockConfig:
    hidden_size = 32

class TestEntropyGating(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.gate = EntropyGating(self.config)
        
    def test_gating_range(self):
        """Test lambda is in [0, 1]"""
        B, T, D = 2, 5, 32
        hidden = torch.randn(B, T, D)
        entropy = torch.rand(B, T) # 0 to 1
        curvature = torch.randn(B, T) # -1 to 1
        
        lam = self.gate(hidden, entropy, curvature)
        
        self.assertEqual(lam.shape, (B, T, 1))
        self.assertTrue(torch.all(lam >= 0.0))
        self.assertTrue(torch.all(lam <= 1.0))

    def test_broadcasting(self):
        """Test scalar entropy broadcasting"""
        B, T, D = 2, 5, 32
        hidden = torch.randn(B, T, D)
        entropy = torch.tensor([0.1, 0.9]) # (B,)
        curvature = torch.tensor([-0.5, -1.5]) # (B,)
        
        lam = self.gate(hidden, entropy, curvature)
        self.assertEqual(lam.shape, (B, T, 1))

if __name__ == '__main__':
    unittest.main()
