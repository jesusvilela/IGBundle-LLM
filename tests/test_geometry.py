import torch
import unittest
import sys
import os
from types import SimpleNamespace

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from igbundle.modules.ops import (
    compute_affinity_matrix, compute_poincare_distance, project_to_poincare_ball,
)
from igbundle.modules.adapter import IGBundleAdapter

torch.set_num_threads(1)

class TestGeometry(unittest.TestCase):
    def test_poincare_distance_stability(self):
        """Test Poincaré distance numerical stability near boundary."""
        B, T, P, D = 1, 1, 2, 4
        
        # Case 1: Identical points (Distance should be 0)
        x = torch.zeros(B, T, P, D)
        y = torch.zeros(B, T, P, D)
        dist = compute_poincare_distance(x, y)
        self.assertTrue(torch.equal(dist, torch.zeros_like(dist)))
        
        # Case 2: Boundary points (Norm close to 1)
        # Without clamping, this would NaN
        x = torch.ones(B, T, P, D) * 0.99 
        # Normalize to be just inside boundary
        x = x / x.norm(dim=-1, keepdim=True) * 0.9999
        y = torch.zeros(B, T, P, D)
        
        dist = compute_poincare_distance(x, y)
        self.assertFalse(torch.isnan(dist).any(), "NaN detected in Poincare distance at boundary")
        
    def test_origin_distance_matches_analytic_formula(self):
        radii = torch.tensor([0.0, 1e-8, 0.1, 0.8, 0.9999], dtype=torch.float64)
        x = torch.stack((radii, torch.zeros_like(radii)), dim=-1)
        distances = compute_poincare_distance(x, torch.zeros(1, 2, dtype=x.dtype))[:, 0]
        torch.testing.assert_close(distances, 2 * torch.atanh(radii), rtol=1e-10, atol=1e-12)

    def test_metric_axioms_on_bounded_points(self):
        generator = torch.Generator().manual_seed(7)
        x = project_to_poincare_ball(torch.randn(12, 4, generator=generator, dtype=torch.float64))
        d = compute_poincare_distance(x, x)
        self.assertTrue(torch.equal(d.diagonal(), torch.zeros(12, dtype=d.dtype)))
        torch.testing.assert_close(d, d.T, rtol=0, atol=1e-12)
        self.assertTrue((d >= 0).all())
        self.assertTrue((d[:, None, :] <= d[:, :, None] + d[None, :, :] + 1e-12).all())

    def test_projection_and_gradients(self):
        x = torch.tensor([[0.0, 0.0], [0.1, -0.2], [3.0, 4.0]],
                         dtype=torch.float64, requires_grad=True)
        projected = project_to_poincare_ball(x)
        torch.testing.assert_close(projected[:2], x[:2])
        self.assertTrue((projected.norm(dim=-1) <= 0.95 + 1e-15).all())
        compute_poincare_distance(projected, projected).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_distance_gradcheck_and_pairwise_shape(self):
        x = torch.tensor([[0.1, 0.2], [-0.2, 0.3]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0.3, 0.1]], dtype=torch.float64, requires_grad=True)
        self.assertEqual(compute_poincare_distance(x, y).shape, (2, 1))
        self.assertTrue(torch.autograd.gradcheck(compute_poincare_distance, (x, y)))

    def test_invalid_points_fail_explicitly(self):
        for value in (1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_poincare_distance(torch.tensor([[value, 0.0]]), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            project_to_poincare_ball(torch.ones(1, 2), max_norm=1.0)

    def test_affinity_large_coordinates_and_half_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            means = torch.full((1, 2, 3, 4), 5.0, dtype=dtype, requires_grad=True)
            affinity = compute_affinity_matrix(means, torch.zeros_like(means),
                                               torch.zeros(1, 2, 3, 3, dtype=dtype))
            self.assertTrue(torch.isfinite(affinity).all())
            torch.testing.assert_close(affinity.sum(-1), torch.ones(1, 2, 3, dtype=dtype))
            affinity.sum().backward()
            self.assertTrue(torch.isfinite(means.grad).all())

    def test_unknown_geometry_is_not_silently_hyperbolic(self):
        with self.assertRaises(ValueError):
            compute_affinity_matrix(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3),
                                    torch.zeros(1, 1, 2, 2), geometry="typo")

    def test_adapter_identity_initialization_and_backward(self):
        config = SimpleNamespace(hidden_size=16, bottleneck_dim=8, num_components=3,
                                 num_categories=4, latent_dim=4, alpha=1.0, beta=1.0,
                                 eta_f=0.1, eta_b=0.1, eta_w=0.1, adapter_scale=0.1,
                                 dropout=0.0, geometry="riemannian")
        for dtype in (torch.float32, torch.bfloat16):
            adapter = IGBundleAdapter(config).to(dtype=dtype)
            hidden = torch.randn(1, 3, 16, dtype=dtype, requires_grad=True)
            output, state = adapter(hidden)
            self.assertTrue(torch.equal(output, hidden))
            self.assertEqual(state.m.shape, (1, 3, 3, 4))
            output.float().square().mean().backward()
            self.assertTrue(torch.isfinite(hidden.grad).all())
            self.assertTrue(torch.isfinite(adapter.out_proj.weight.grad).all())
            self.assertGreater(adapter.out_proj.weight.grad.abs().sum().item(), 0)

if __name__ == "__main__":
    unittest.main()
