import torch
import torch.nn as nn
import unittest
import sys
import os
sys.path.append(os.path.abspath("src"))

from igbundle.dynamics.hamiltonian import HamiltonianSystem
from igbundle.geometry.poincare import PoincareBall
from igbundle.dynamics.potential import NeuralPotential
from igbundle.cognition.meta import MetaCognitiveLoop

class TestMetaCognition(unittest.TestCase):
    def setUp(self):
        self.D = 16
        self.manifold = PoincareBall(dim=self.D)
        self.potential = NeuralPotential(latent_dim=self.D, hidden_dim=32)
        self.vf = HamiltonianSystem(self.manifold, potential_module=self.potential)
        self.meta = MetaCognitiveLoop(self.vf, threshold=0.1, refine_steps=10, lr=0.1)
        
    def test_verify_energy(self):
        """Test energy calculation."""
        q = torch.randn(1, 1, 1, self.D) * 0.1
        v, valid = self.meta.verify(q)
        print(f"Initial Energy: {v.item()}")
        self.assertEqual(v.shape, (1, 1, 1))
        
    def test_refine_logic(self):
        """Test that refinement decreases potential energy."""
        # Create a "High Energy" point (far from origin, if potential is initialized to favor origin?)
        # Random initialization of potential might not favor origin.
        # But gradient descent should find a local minimum.
        
        q = torch.randn(1, 1, 1, self.D) * 0.5
        v_init, _ = self.meta.verify(q)
        print(f"Energy before: {v_init.item()}")
        
        q_refined, trace = self.meta.refine(q)
        v_final = trace[-1]
        print(f"Energy after: {v_final}")
        
        self.assertLess(v_final, v_init.item() + 1e-5, "Energy should decrease or stay same")
        
    def test_forward_pass(self):
        """Test the full loop integration."""
        q = torch.randn(1, 1, 1, self.D) * 0.5
        # Force threshold low to trigger refine
        self.meta.threshold = -100.0 
        
        q_out, info = self.meta(q)
        
        self.assertTrue(info['refined'])
        self.assertIn('trace', info)
        self.assertEqual(len(info['trace']), 11) # 10 steps + final check

if __name__ == '__main__':
    unittest.main()
