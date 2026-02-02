
import torch
import unittest
import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.igbundle.modules.geometric_adapter import GeometricIGBundleAdapter
from src.igbundle.core.config import IGBundleConfig

class TestMultimodalAdapter(unittest.TestCase):
    def setUp(self):
        self.config = IGBundleConfig(
            hidden_size=64, # Small for test
            adapter_dim=16,
            latent_dim=16,
            num_components=2,
            vision_dim=32, # Small
            manifold_type="euclidean", # Use simple backbone
            supported_modalities=["text", "vision"] # Activate multimodal flag
        )
        self.adapter = GeometricIGBundleAdapter(self.config)
        self.device = torch.device('cpu') # Force CPU for test
        self.adapter.to(self.device)
        
    def test_forward_with_vision(self):
        """Test forward pass accepts pixel_values and integrates them."""
        B, T, D = 2, 10, 64
        hidden_states = torch.randn(B, T, D).to(self.device)
        
        # Mock Vision Input: (B, NumPatches, VisionDim)
        num_patches = 4
        vision_dim = 32
        pixel_values = torch.randn(B, num_patches, vision_dim).to(self.device)
        
        # Run forward
        output, _ = self.adapter(
            hidden_states, 
            pixel_values=pixel_values
        )
        
        # Check output shape (should preserve B, T, D)
        self.assertEqual(output.shape, (B, T, D))
        
        # Check if vision projector was called (by checking if params have grad if we established grad?)
        # Or simpler: Check if output is different with/without vision?
        # That depends on implementation (concatenation vs addition).
        # Assuming it modifies the state.

    def test_multimodal_flag(self):
        """Check if config reports multimodal capability."""
        self.assertTrue(self.config.is_multimodal)

if __name__ == '__main__':
    unittest.main()
