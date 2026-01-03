import torch
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from igbundle.modules.ops import compute_poincare_distance
from igbundle.geometry.riemannian import RiemannianGeometry

class TestGeometry(unittest.TestCase):
    def test_poincare_distance_stability(self):
        """Test Poincaré distance numerical stability near boundary."""
        B, T, P, D = 1, 1, 2, 4
        
        # Case 1: Identical points (Distance should be 0)
        # Note: Epsilon clamping in ops.py (min=1+1e-5) introduces small bias: acosh(1.00001) ~ 0.0045
        x = torch.zeros(B, T, P, D)
        y = torch.zeros(B, T, P, D)
        dist = compute_poincare_distance(x, y)
        self.assertTrue(torch.allclose(dist, torch.zeros_like(dist), atol=1e-2), 
                       f"Distance at identity too large: {dist.max().item()}")
        
        # Case 2: Boundary points (Norm close to 1)
        # Without clamping, this would NaN
        x = torch.ones(B, T, P, D) * 0.99 
        # Normalize to be just inside boundary
        x = x / x.norm(dim=-1, keepdim=True) * 0.9999
        y = torch.zeros(B, T, P, D)
        
        dist = compute_poincare_distance(x, y)
        self.assertFalse(torch.isnan(dist).any(), "NaN detected in Poincare distance at boundary")
        
    def test_curvatures(self):
        """Test curvature consistency."""
        # TODO: Implement with AutoDiff refactor
        pass

if __name__ == "__main__":
    unittest.main()
