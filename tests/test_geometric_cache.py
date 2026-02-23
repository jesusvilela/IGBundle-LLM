
import torch
import torch.nn as nn
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from igbundle.modules.geometric_cache import GeometricCacheManager, GeometricKVCache
from igbundle.geometry.hyperbolic import PoincareBall

class MockConfig:
    num_hidden_layers = 2
    manifold_curvature = 1.0

class MockProjector(nn.Module):
    def forward(self, x):
        return PoincareBall.from_euclidean(x, c=1.0)

class TestGeometricKVCache(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()
        self.manager = GeometricCacheManager(self.config)
        self.projector = MockProjector()

    def test_update_layer(self):
        # B=1, H=2, L=5, D=4
        new_k = torch.randn(1, 2, 5, 4)
        new_v = torch.randn(1, 2, 5, 4)
        
        # First update
        self.manager.update(0, new_k, new_v, self.projector)
        state = self.manager.get_layer(0)
        
        self.assertIsNotNone(state.key_cache_euclidean)
        self.assertIsNotNone(state.key_cache_manifold)
        self.assertEqual(state.key_cache_manifold.shape, (1, 2, 5, 4))
        
        # Second update (append)
        new_k2 = torch.randn(1, 2, 3, 4)
        new_v2 = torch.randn(1, 2, 3, 4)
        self.manager.update(0, new_k2, new_v2, self.projector)
        state = self.manager.get_layer(0)
        
        self.assertEqual(state.key_cache_euclidean.shape[2], 8) # 5 + 3
        self.assertEqual(state.key_cache_manifold.shape[2], 8)
        
        # Check manifold projection
        # last 3 tokens of manifold cache should mimic projection of new_k2
        expected_proj = self.projector(new_k2)
        self.assertTrue(torch.allclose(state.key_cache_manifold[:,:,5:,:], expected_proj))

if __name__ == '__main__':
    unittest.main()
