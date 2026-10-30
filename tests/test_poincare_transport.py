"""Bounded checks against independent geometry, not a trained-model result."""

import json
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from igbundle.geometry.poincare import PoincareBall
from scripts.transport_reference import lorentz_transport, witness

torch.set_num_threads(1)


class PoincareTransportTests(unittest.TestCase):
    def samples(self, dim, c):
        generator = torch.Generator().manual_seed(41)
        x, y = [torch.randn(12, dim, generator=generator, dtype=torch.float64) for _ in range(2)]
        x = .6 * x / x.norm(dim=-1, keepdim=True) / math.sqrt(c)
        y = .4 * y / y.norm(dim=-1, keepdim=True) / math.sqrt(c)
        v, w = [torch.randn(12, dim, generator=generator, dtype=torch.float64) for _ in range(2)]
        return x, y, v, w

    def test_independent_lorentz_transport_across_curvatures_and_dimensions(self):
        for dim in (2, 3, 8):
            for c in (.25, 1., 4.):
                with self.subTest(dim=dim, c=c):
                    x, y, v, _ = self.samples(dim, c)
                    actual = PoincareBall(dim, c).parallel_transport(x, y, v)
                    torch.testing.assert_close(actual, lorentz_transport(x, y, v, c),
                                               rtol=1e-12, atol=1e-12)

    def test_metric_products_and_reverse_transport(self):
        for c in (.25, 1., 4.):
            x, y, v, w = self.samples(3, c)
            ball = PoincareBall(3, c)
            pv, pw = [ball.parallel_transport(x, y, value) for value in (v, w)]
            before = ball.conformal_factor(x).square() * (v * w).sum(-1, keepdim=True)
            after = ball.conformal_factor(y).square() * (pv * pw).sum(-1, keepdim=True)
            torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(ball.parallel_transport(y, x, pv), v, rtol=1e-12, atol=1e-12)

    def test_cotangent_pairing_and_kinetic_energy(self):
        x, y, v, p = self.samples(3, 4.)
        ball = PoincareBall(3, 4.)
        transported_v = ball.parallel_transport(x, y, v)
        transported_p = ball.cotangent_transport(x, y, p)
        torch.testing.assert_close((transported_p * transported_v).sum(-1), (p * v).sum(-1),
                                   rtol=1e-12, atol=1e-12)
        before = p.square().sum(-1, keepdim=True) / (2 * ball.conformal_factor(x).square())
        after = transported_p.square().sum(-1, keepdim=True) / (2 * ball.conformal_factor(y).square())
        torch.testing.assert_close(after, before, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ball.cotangent_transport(y, x, transported_p), p,
                                   rtol=1e-12, atol=1e-12)
        wrong_p = ball.parallel_transport(x, y, p)
        wrong_energy = wrong_p.square().sum(-1, keepdim=True) / (2 * ball.conformal_factor(y).square())
        self.assertGreater((wrong_energy - before).abs().max().item(), .01)

    def test_radial_and_stationary_paths(self):
        ball = PoincareBall(2)
        origin = torch.zeros(2, dtype=torch.float64)
        point = torch.tensor([.4, 0.], dtype=torch.float64)
        v = torch.tensor([.1, .3], dtype=torch.float64)
        torch.testing.assert_close(ball.parallel_transport(point, point, v), v, rtol=0, atol=1e-15)
        torch.testing.assert_close(ball.parallel_transport(origin, point, v), .84 * v)
        torch.testing.assert_close(ball.cotangent_transport(origin, point, v), v / .84)

    def test_triangle_holonomy_and_scale_only_counterexample(self):
        result = witness()
        self.assertGreater(abs(result["signed_holonomy_radians"]), .23)
        self.assertLess(result["angle_error"], 1e-12)
        self.assertEqual(result["scale_only_signed_holonomy_radians"], 0.)
        # Angle defect from independent hyperboloid tangents, not gyration.
        a, b = .3, .4
        A = torch.tensor([(1+a*a)/(1-a*a), 2*a/(1-a*a), 0.], dtype=torch.float64)
        B = torch.tensor([(1+b*b)/(1-b*b), 0., 2*b/(1-b*b)], dtype=torch.float64)
        O = torch.tensor([1., 0., 0.], dtype=torch.float64)

        def dot(u, v):
            return -u[0]*v[0] + (u[1:]*v[1:]).sum()

        def angle(at, left, right):
            u, v = [point + dot(at, point) * at for point in (left, right)]
            return math.acos((dot(u, v) / (dot(u, u)*dot(v, v)).sqrt()).item())

        defect = math.pi - angle(O, A, B) - angle(A, O, B) - angle(B, A, O)
        self.assertAlmostEqual(result["signed_holonomy_radians"], -defect, places=12)

    def test_published_witness_matches_current_reference(self):
        saved = json.loads((ROOT / "docs" / "benchmark-results" / "transport-reference.json")
                           .read_text(encoding="utf-8"))
        actual = witness()
        self.assertEqual(set(saved), set(actual))
        for key, value in actual.items():
            with self.subTest(key=key):
                if isinstance(value, (float, list)):
                    torch.testing.assert_close(torch.tensor(saved[key], dtype=torch.float64),
                                               torch.tensor(value, dtype=torch.float64),
                                               rtol=0, atol=1e-12)
                else:
                    self.assertEqual(saved[key], value)

    def test_broadcasting_and_low_precision_promotion(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.zeros(2, 1, 3, dtype=dtype)
            y = torch.full((1, 4, 3), .125, dtype=dtype)
            v = torch.ones(3, dtype=dtype)
            actual = PoincareBall(3).parallel_transport(x, y, v)
            self.assertEqual(actual.shape, (2, 4, 3))
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.isfinite(actual).all())

    def test_gradients_including_origin(self):
        ball = PoincareBall(2)
        for point in ([.1, -.2], [0., 0.]):
            x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            y = torch.tensor([.3, .1], dtype=torch.float64, requires_grad=True)
            v = torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True)
            for operation in (ball.parallel_transport, ball.cotangent_transport):
                self.assertTrue(torch.autograd.gradcheck(operation, (x, y, v)))

    def test_invalid_domains_fail_instead_of_clipping(self):
        for c in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(c=c), self.assertRaises(ValueError):
                PoincareBall(2, c)
        ball = PoincareBall(2, 4.)
        good = torch.zeros(2)
        for point in (torch.tensor([.5, 0.]), torch.tensor([.6, 0.]),
                      torch.tensor([float("nan"), 0.]), torch.tensor([float("inf"), 0.])):
            with self.assertRaises(ValueError):
                ball.parallel_transport(good, point, good)
        for bad in (torch.zeros(3), torch.zeros(2, dtype=torch.int64), torch.zeros(2).double()):
            with self.assertRaises(ValueError):
                ball.parallel_transport(good, good, bad)
        with self.assertRaises(ValueError):
            ball.parallel_transport(torch.zeros(2, 2), torch.zeros(3, 2), good)
        with self.assertRaises(ValueError):
            ball.parallel_transport(good, good, torch.tensor([float("inf"), 0.]))


if __name__ == "__main__":
    unittest.main()
