
import torch
import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from igbundle.geometry.hyperbolic import PoincareBall

class TestPoincareBall(unittest.TestCase):
    def setUp(self):
        self.c = 1.0
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_mobius_add_zero(self):
        """Test x + 0 = x"""
        x = torch.randn(10, 5) * 0.1 # Small norm
        zero = torch.zeros_like(x)
        res = PoincareBall.mobius_add(x, zero, self.c)
        self.assertTrue(torch.allclose(res, x, atol=1e-5))
        
        res2 = PoincareBall.mobius_add(zero, x, self.c)
        self.assertTrue(torch.allclose(res2, x, atol=1e-5))

    def test_dist_positivity(self):
        """Test d(x,y) >= 0 and d(x,x) = 0"""
        x = torch.randn(10, 5) * 0.5
        y = torch.randn(10, 5) * 0.5
        
        # d(x, x)
        d_xx = PoincareBall.dist(x, x, self.c)
        self.assertTrue(torch.allclose(d_xx, torch.zeros_like(d_xx), atol=1e-4))
        
        # d(x, y) > 0
        d_xy = PoincareBall.dist(x, y, self.c)
        self.assertTrue(torch.all(d_xy >= -1e-6)) # Allow tiny float error

    def test_exp_log_consistency(self):
        """Test Exp_x(Log_x(y)) approx y"""
        # Pick x, y relatively close to origin to avoid numerical issues
        x = torch.randn(5, 5) * 0.3
        y = torch.randn(5, 5) * 0.3
        
        v = PoincareBall.log_map(x, y, self.c)
        y_recon = PoincareBall.exp_map(x, v, self.c)
        
        diff = torch.norm(y - y_recon, dim=-1)
        print(f"Exp/Log Diff: {diff.mean().item()}")
        
        # Atol 1e-4 is acceptable for float32 hyperbolic ops
        self.assertTrue(torch.allclose(y, y_recon, atol=1e-4))

    def test_project(self):
        """Test projection constraints"""
        # Create vectors outside ball
        x = torch.randn(10, 5) * 5.0 
        x_proj = PoincareBall.project(x, self.c)
        
        norm = torch.norm(x_proj, dim=-1)
        max_norm = 1.0 / (self.c ** 0.5)
        
        self.assertTrue(torch.all(norm < max_norm + 1e-5))

if __name__ == '__main__':
    unittest.main()
