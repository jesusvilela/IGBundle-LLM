import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from igbundle.modules.kl import kl_diag_gauss, kl_categorical_logits
from igbundle.modules.state import MixtureState

class TestIGBundle(unittest.TestCase):
    def test_kl_diag_gauss(self):
        B, T, P, D = 2, 5, 4, 8
        m1 = torch.randn(B, T, P, 1, D)
        ls1 = torch.randn(B, T, P, 1, D)
        m2 = torch.randn(B, T, 1, P, D)
        ls2 = torch.randn(B, T, 1, P, D)
        
        # Test output shape
        kl = kl_diag_gauss(m1, ls1, m2, ls2)
        self.assertEqual(kl.shape, (B, T, P, P))
        
        # Test identity
        m = torch.randn(D)
        ls = torch.randn(D)
        kl_val = kl_diag_gauss(m, ls, m, ls)
        self.assertTrue(torch.allclose(kl_val, torch.zeros_like(kl_val), atol=1e-5))
        
    def test_kl_categorical(self):
        B, T, P, K = 2, 5, 4, 10
        u1 = torch.randn(B, T, P, 1, K)
        u2 = torch.randn(B, T, 1, P, K)
        
        kl = kl_categorical_logits(u1, u2)
        self.assertEqual(kl.shape, (B, T, P, P))
        
        # Identity
        u = torch.randn(K)
        kl_val = kl_categorical_logits(u, u)
        self.assertTrue(torch.allclose(kl_val, torch.zeros_like(kl_val), atol=1e-5))
        
    def test_mixture_state(self):
        B, T, P, D, K = 2, 5, 3, 4, 6
        w = torch.randn(B, T, P)
        m = torch.randn(B, T, P, D)
        ls = torch.randn(B, T, P, D)
        u = torch.randn(B, T, P, K)
        
        s = MixtureState(w, m, ls, u)
        
        self.assertTrue(torch.is_tensor(s.sigma))
        self.assertTrue(torch.is_tensor(s.p))
        self.assertEqual(s.sigma.shape, (B, T, P, D))

if __name__ == '__main__':
    unittest.main()
