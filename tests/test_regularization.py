
import torch
import torch.nn as nn
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from igbundle.modules.regularization import LipschitzPenalty, spectral_normalize_module
from igbundle.geometry.hyperbolic import PoincareBall

class MockProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        self.c = 1.0
        
    def forward(self, x):
        out = self.net(x)
        # Project to ball
        return PoincareBall.from_euclidean(out, self.c)

class TestRegularization(unittest.TestCase):
    def setUp(self):
        self.projector = MockProjector()
        torch.manual_seed(42)

    def test_spectral_norm_application(self):
        """Test if spectral norm hook is registered."""
        spectral_normalize_module(self.projector)
        
        has_sn = False
        for name, module in self.projector.named_modules():
            if isinstance(module, nn.Linear):
                if 'weight_orig' in module._parameters:
                    has_sn = True
        
        self.assertTrue(has_sn, "Spectral Norm not applied")

    def test_lipschitz_penalty_computation(self):
        """Test penalty calculation."""
        x = torch.randn(5, 10)
        penalty = LipschitzPenalty.compute_penalty(self.projector, x, c=1.0, k=0.1)
        
        # k=0.1 is very strict, so we expect penalty > 0 if the net is random
        print(f"Penalty: {penalty.item()}")
        self.assertGreater(penalty.item(), 0.0)
        
        # Test zero penalty for loose k
        penalty_loose = LipschitzPenalty.compute_penalty(self.projector, x, c=1.0, k=100.0)
        self.assertEqual(penalty_loose.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
