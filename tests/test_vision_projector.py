
import torch
import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.igbundle.modules.vision import VisionProjector

class TestVisionProjector(unittest.TestCase):
    def setUp(self):
        self.vision_dim = 1152 # SigLIP
        self.bottleneck_dim = 256
        self.dropout = 0.0
        self.projector = VisionProjector(self.vision_dim, self.bottleneck_dim, self.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projector.to(self.device)
        
    def test_shape_consistency(self):
        """Verify input (B, N, V) -> output (B, N, D_bot)"""
        B, N = 2, 16
        x = torch.randn(B, N, self.vision_dim).to(self.device)
        out = self.projector(x)
        
        self.assertEqual(out.shape, (B, N, self.bottleneck_dim))
        
    def test_gradient_flow(self):
        """Verify gradients flow through the gated mechanism"""
        B, N = 1, 4
        x = torch.randn(B, N, self.vision_dim).to(self.device)
        x.requires_grad = True
        
        out = self.projector(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertNotEqual(x.grad.abs().sum().item(), 0.0)
        
        # Check bias gradient
        self.assertIsNotNone(self.projector.visual_bias.grad)

    def test_gating_mechanism(self):
        """Check if output is bounded by layer norm / gate nature?"""
        # Hard to check specific values, but we can check if it runs without NaNs
        B, N = 2, 10
        x = torch.randn(B, N, self.vision_dim).to(self.device) * 100 # Large input
        out = self.projector(x)
        
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

if __name__ == '__main__':
    unittest.main()
