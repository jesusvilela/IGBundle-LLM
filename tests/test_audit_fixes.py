"""Regression tests for the 2026-07 external-audit fixes.

Pins the four code-level defects identified by the two external audit passes:
  1. ops.py projected means to the ball via element-wise tanh (norm up to sqrt(D)).
  2. poincare.py clamped ||(-x)(+)y|| at 1-eps instead of clamping sqrt(c)*||.||
     (wrong ball radius for c != 1).
  3. geometry.sectional_curvature was removed but bundle_curvature_loss /
     adaptive_curvature_loss still called it -> AttributeError.
  4. The curvature estimator was weight-invariant (hardcoded conformal factor)
     and used an uncalibrated formula; now corrected and optionally trainable
     via config.learnable_conformal.
"""
import math
import os
import sys
import unittest

import torch

_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from igbundle.core.config import IGBundleConfig
from igbundle.geometry.poincare import PoincareBall
from igbundle.geometry.riemannian import RiemannianGeometry, bundle_curvature_loss
from igbundle.modules.ops import compute_affinity_matrix


def _make_config(**overrides):
    defaults = dict(latent_dim=8, num_components=2, num_categories=4,
                    manifold_type="riemannian")
    defaults.update(overrides)
    return IGBundleConfig(**defaults)


class TestPoincareCurvatureClamp(unittest.TestCase):
    """Audit item: atanh domain guard must scale with the ball radius 1/sqrt(c)."""

    def test_distance_matches_closed_form_for_c_quarter(self):
        c = 0.25  # ball radius 1/sqrt(c) = 2
        ball = PoincareBall(dim=4, c=c)
        x = torch.zeros(1, 4)
        y = torch.zeros(1, 4)
        y[0, 0] = 1.5  # valid point: inside radius-2 ball, outside unit ball
        # d(0, y) = (2/sqrt(c)) * atanh(sqrt(c) * ||y||)
        expected = (2.0 / math.sqrt(c)) * math.atanh(math.sqrt(c) * 1.5)
        got = ball.distance(x, y).item()
        self.assertAlmostEqual(got, expected, places=4)

    def test_log_map_norm_matches_distance_at_origin(self):
        c = 0.25
        ball = PoincareBall(dim=4, c=c)
        x = torch.zeros(1, 4)
        y = torch.zeros(1, 4)
        y[0, 0] = 1.5
        # Ganea convention: the RIEMANNIAN norm of log_x(y) equals d(x, y).
        # At the origin lambda_0 = 2/(1 - c*0) = 2, so
        # d(0, y) = lambda_0 * ||log_0(y)||_Euclidean.
        log_norm_euc = ball.log_map(x, y).norm().item()
        lambda_0 = ball.conformal_factor(x).item()
        dist = ball.distance(x, y).item()
        self.assertAlmostEqual(lambda_0 * log_norm_euc, dist, places=4)


class TestOpsBallProjection(unittest.TestCase):
    """Audit item: affinity path must project by vector norm, not element-wise tanh."""

    def test_riemannian_affinity_stays_finite_and_normalized(self):
        B, T, P, D = 1, 2, 3, 16
        means = torch.randn(B, T, P, D) * 10.0  # large norms on purpose
        log_sigmas = torch.zeros(B, T, P, D)
        d_fiber = torch.rand(B, T, P, P)
        A = compute_affinity_matrix(means, log_sigmas, d_fiber, geometry="riemannian")
        self.assertTrue(torch.isfinite(A).all())
        self.assertTrue(torch.allclose(A.sum(dim=-1), torch.ones(B, T, P), atol=1e-5))


class TestCurvatureEstimator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_sectional_curvature_compat_shim_exists(self):
        geo = RiemannianGeometry(_make_config())
        pos = torch.randn(1, 2, 2, 8) * 0.3
        k = geo.sectional_curvature(pos)
        self.assertEqual(k.shape, (1, 2, 2))
        self.assertTrue(torch.isfinite(k).all())

    def test_bundle_curvature_loss_no_attribute_error(self):
        geo = RiemannianGeometry(_make_config())
        pos = torch.randn(1, 2, 2, 8) * 0.3
        loss = bundle_curvature_loss(geo, pos, target_curvature=-1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_curvature_is_zero_when_conformal_scale_zero(self):
        """With lambda == 1 the metric is position-constant => flat."""
        geo = RiemannianGeometry(_make_config(learnable_conformal=True))
        with torch.no_grad():
            geo.conformal_scale.zero_()
        pos = torch.randn(1, 2, 2, 8) * 0.5
        k = geo.estimate_sectional_curvature_stochastic(pos, num_samples=4)
        self.assertTrue(torch.allclose(k, torch.zeros_like(k), atol=1e-4))

    def test_curvature_responds_to_conformal_scale(self):
        """The audit's core finding: kappa was weight-invariant. It must now
        respond to the (trainable) conformal scale."""
        cfg = _make_config(learnable_conformal=True)
        geo = RiemannianGeometry(cfg)
        pos = torch.randn(1, 2, 2, 8) * 0.5
        torch.manual_seed(42)
        k_half = geo.estimate_sectional_curvature_stochastic(pos, num_samples=8)
        with torch.no_grad():
            geo.conformal_scale.fill_(0.05)
        torch.manual_seed(42)
        k_small = geo.estimate_sectional_curvature_stochastic(pos, num_samples=8)
        self.assertFalse(torch.allclose(k_half, k_small, atol=1e-6))

    def test_learnable_conformal_flag_controls_parameterization(self):
        geo_fixed = RiemannianGeometry(_make_config())
        geo_learn = RiemannianGeometry(_make_config(learnable_conformal=True))
        fixed_params = {n for n, _ in geo_fixed.named_parameters()}
        learn_params = {n for n, _ in geo_learn.named_parameters()}
        self.assertNotIn("conformal_scale", fixed_params)  # checkpoint-compatible
        self.assertIn("conformal_scale", learn_params)
        # Non-persistent buffer => state_dict unchanged for the default path
        self.assertNotIn("conformal_scale", geo_fixed.state_dict())

    def test_log_map_flat_fast_path(self):
        geo = RiemannianGeometry(_make_config())
        base = torch.randn(1, 1, 2, 8) * 0.2
        target = torch.randn(1, 1, 2, 8) * 0.2
        metric = geo.get_metric(base)
        v = geo.log_map(base, target, metric)
        self.assertTrue(torch.allclose(v, target - base))


class TestConfigFlags(unittest.TestCase):
    def test_audit_flags_default_off(self):
        cfg = _make_config()
        self.assertFalse(cfg.learnable_conformal)
        self.assertFalse(cfg.differentiable_dynamics)


if __name__ == "__main__":
    unittest.main()
