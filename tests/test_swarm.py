
import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from igbundle.fibers.swarm import MultiFiberExecutor
from igbundle.geometry.hyperbolic import PoincareBall

class MockConfig:
    num_components = 3
    latent_dim = 4
    manifold_curvature = 1.0

class TestFiberSwarm(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.swarm = MultiFiberExecutor(self.config)
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_consensus_averaging(self):
        """Test if consensus finds the 'middle'."""
        # Fiber 0: At origin
        # Fiber 1: At (0.5, 0, ...)
        # Fiber 2: At (-0.5, 0, ...)
        # Equal weights -> Should be near origin
        
        B = 1
        D = 4
        proposals = torch.zeros(B, 3, D).double()
        proposals[0, 1, 0] = 0.5
        proposals[0, 2, 0] = -0.5
        
        weights = torch.tensor([[1.0, 1.0, 1.0]]).double()
        
        cons = self.swarm.consensus(proposals, weights)
        
        print(f"Consensus: {cons}")
        self.assertTrue(torch.norm(cons) < 0.1)

if __name__ == '__main__':
    unittest.main()
