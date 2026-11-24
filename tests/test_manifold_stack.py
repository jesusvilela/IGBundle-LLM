import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from igbundle.telemetry import format_stack_panel, summarize_manifold_stack


class DummyConfig:
    use_dynamics = True
    use_delta_fiber = True


class DummyAdapter:
    H = 5
    D = 3
    P = 2
    K = 4
    D_bot = 2
    manifold_type = "poincare"
    cfg = DummyConfig()
    use_delta_fiber = True


class DummyState:
    def __init__(self, coords, fibers):
        self.base_coordinates = coords
        self.fiber_sections = fibers
        self.active_indices = torch.tensor([[0, 1]])
        self.consensus_loss = torch.tensor(0.2)
        self.meta_info = {"final_energy": 0.25}


class TestManifoldStackTelemetry(unittest.TestCase):
    def test_summary_keeps_full_latent_dimension(self):
        coords = torch.tensor(
            [[
                [[0.10, 0.20, 0.30], [0.20, 0.10, 0.10]],
                [[0.40, 0.10, 0.20], [0.10, 0.30, 0.20]],
            ]],
            dtype=torch.float32,
        )
        logits = torch.randn(1, 2, 2, 4)
        fibers = torch.softmax(logits, dim=-1)
        hidden = torch.ones(1, 2, 5)
        previous_hidden = torch.zeros(1, 5)
        previous_geo = coords[:, 0, 0, :]

        summary = summarize_manifold_stack(
            DummyAdapter(),
            DummyState(coords, fibers),
            hidden_state=hidden,
            previous_geo=previous_geo,
            previous_hidden=previous_hidden,
            constraint_score=0.75,
        )

        self.assertEqual(summary["dims"]["latent_dim"], 3)
        self.assertEqual(summary["dims"]["components"], 2)
        self.assertEqual(summary["dims"]["categories"], 4)
        self.assertGreater(summary["radius_max"], abs(summary["projection"][0]))
        self.assertGreater(summary["geodesic_step"], 0.0)
        self.assertGreater(summary["euclidean_step"], 0.0)
        self.assertIn("mutual_recognition", summary["quality_vector"])
        self.assertIn("self_reflection", summary["quality_vector"])
        self.assertIn("constraint_alignment", summary["quality_vector"])
        self.assertAlmostEqual(summary["quality_vector"]["constraint_alignment"], 0.75)

        panel = format_stack_panel(summary)
        self.assertIn("H=5 D=3 P=2 K=4", panel)
        self.assertIn("Lholo=", panel)
        self.assertIn("SELF=", panel)

    def test_one_dimensional_projection_is_valid_view(self):
        coords = torch.tensor([[[[0.4]]]], dtype=torch.float32)
        fibers = torch.softmax(torch.randn(1, 1, 1, 2), dim=-1)

        summary = summarize_manifold_stack(DummyAdapter(), DummyState(coords, fibers))

        self.assertEqual(summary["dims"]["latent_dim"], 1)
        self.assertAlmostEqual(summary["projection"][0], 0.4, places=6)
        self.assertEqual(summary["projection"][1], 0.0)


if __name__ == "__main__":
    unittest.main()