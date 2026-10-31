"""Tiny legacy-adapter execution checks, not geometric-law certification."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter

torch.set_num_threads(1)


def make_adapter(*, dtype=torch.float64, nonzero=False, components=2):
    torch.manual_seed(19)
    config = SimpleNamespace(
        hidden_size=12, bottleneck_dim=6, latent_dim=3, num_categories=2,
        num_components=components, num_sheaf_patches=2, adapter_scale=.1,
        eta_b=.1, eta_f=.1, dropout=0.,
    )
    model = GeometricIGBundleAdapter(config).to(dtype=dtype).eval()
    if nonzero:
        with torch.no_grad():
            weight = model.output_proj.weight
            weight.copy_(torch.linspace(-.02, .02, weight.numel(), dtype=dtype)
                         .reshape_as(weight))
    return model


def inputs(dtype=torch.float64):
    return torch.linspace(-.3, .4, 2 * 4 * 12, dtype=dtype).reshape(2, 4, 12)


class LegacyGeometricAdapterTests(unittest.TestCase):
    def test_identity_forward_and_state_shapes(self):
        for dtype in (torch.float32, torch.float64):
            for components in (1, 2):
                with self.subTest(dtype=dtype, components=components):
                    model = make_adapter(dtype=dtype, components=components)
                    self.assertLess(sum(p.numel() for p in model.parameters()), 100_000)
                    x = inputs(dtype)
                    for length in (1, 4):
                        output, state = model(x[:, :length])
                        self.assertTrue(torch.equal(output, x[:, :length]))
                        self.assertEqual(state.base_coordinates.shape, (2, length, components, 3))
                        self.assertEqual(state.fiber_sections.shape, (2, length, components, 2))
                        self.assertEqual(state.metric.metric_tensor.shape, (2, length, components, 3, 3))
                        self.assertTrue(torch.isfinite(state.base_coordinates).all())
                        self.assertTrue(torch.isfinite(state.fiber_sections).all())
                        torch.testing.assert_close(state.fiber_sections.sum(-1),
                                                   torch.ones(2, length, components, dtype=dtype))
                        state.mixture_state.assert_shapes()

    def test_smoothing_matches_historical_formula_without_mutation(self):
        model = make_adapter()
        coordinates = torch.linspace(-1., 1., 2 * 4 * 2 * 3, dtype=torch.float64)
        coordinates = coordinates.reshape(2, 4, 2, 3).requires_grad_()
        before = coordinates.detach().clone()
        expected = coordinates.clone()
        for t in range(1, 4):
            for p in range(2):
                current, previous = coordinates[:, t, p], coordinates[:, t - 1, p]
                expected[:, t, p] = current - .1 * (current - previous)
        actual = model._coordinate_smoothing_update(coordinates)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(coordinates.detach(), before))
        torch.testing.assert_close(
            model._coordinate_smoothing_update(coordinates[:, :1]), coordinates[:, :1],
            rtol=0, atol=0,
        )
        self.assertTrue(torch.autograd.gradcheck(model._coordinate_smoothing_update, (coordinates,)))

    def test_nonzero_forward_backward_does_not_use_connection(self):
        model = make_adapter(nonzero=True)
        x = inputs().requires_grad_()
        with patch.object(model.riemannian_geometry, "parallel_transport",
                          side_effect=AssertionError("Forward is not parallel transport")), \
             patch.object(model.riemannian_geometry, "christoffel_symbols",
                          side_effect=AssertionError("Forward does not use the neural connection")):
            output, _ = model(x)
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreater((output - x).abs().max().item(), 1e-6)
            output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ("input_proj.weight", "output_proj.weight", "base_coord_proj.weight",
                     "base_update_net.0.weight", "riemannian_geometry.metric_chol"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertTrue(all(parameter.grad is None
                            for parameter in model.riemannian_geometry.christoffel_net.parameters()))

    def test_strict_tensor_state_roundtrip_reproduces_nonzero_output(self):
        source = make_adapter(nonzero=True)
        target = make_adapter()
        x = inputs()
        with torch.no_grad():
            expected, _ = source(x)
            before, _ = target(x)
        self.assertGreater((expected - before).abs().max().item(), 1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-state.pt"
            torch.save(source.state_dict(), path)
            target.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            actual, _ = target(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(set(source.state_dict()), set(target.state_dict()))

    def test_future_tokens_do_not_change_prefix(self):
        model = make_adapter(nonzero=True)
        x = inputs()
        changed = x.clone()
        changed[:, 2:] += 2
        with torch.no_grad():
            output, _ = model(x)
            other, _ = model(changed)
            prefix, _ = model(x[:, :2])
        torch.testing.assert_close(other[:, :2], output[:, :2], rtol=0, atol=1e-12)
        torch.testing.assert_close(prefix, output[:, :2], rtol=0, atol=1e-12)

    def test_smoothing_needs_previous_coordinate_for_chunked_equivalence(self):
        model = make_adapter()
        coordinates = torch.arange(4 * 2 * 3, dtype=torch.float64).reshape(1, 4, 2, 3)
        full = model._coordinate_smoothing_update(coordinates)
        separate = torch.cat([model._coordinate_smoothing_update(coordinates[:, t:t+1])
                              for t in range(4)], dim=1)
        self.assertGreater((full - separate).abs().max().item(), .5)


if __name__ == "__main__":
    unittest.main()
