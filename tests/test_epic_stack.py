import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from igbundle.telemetry import format_epic_panel, summarize_epic_stack


class DummyAdapter:
    use_dynamics = True
    use_geodesic_attn = True
    use_delta_fiber = True
    vf = object()
    dynamics = object()
    integrator = object()
    geodesic_attn = object()
    compiler = object()
    vision_projector = object()
    phase_memory = object()
    consensus = object()
    delta_fiber_update = object()
    fiber_store = object()
    potential_net = object()
    meta_loop = object()
    manifold_type = "poincare"

    class Executor:
        resonator_state = (torch.zeros(2), torch.zeros(2))

    fiber_executor = Executor()


class DummyState:
    op_logits = torch.ones(1, 2, 3)
    consensus_loss = torch.tensor(0.1)
    active_indices = torch.tensor([[0, 1]])
    meta_info = {"refined": True, "initial_energy": 2.0, "final_energy": 1.0}


class TestEpicStack(unittest.TestCase):
    def test_epic_stack_reports_runtime_surfaces_and_ood(self):
        telemetry = {
            "constraint_score": 0.05,
            "curvature_status": "estimated-riemannian",
            "adapter_provenance": {"mode": "gemma4-teacher-geometry-core", "checkpoint_path": "adapter.pt"},
            "n_stack": {"radius_max": 0.92, "holoportation_ratio": 9.5, "sheaf_defect": 0.2, "projection": [0.1, 0.2]},
        }

        summary = summarize_epic_stack(DummyAdapter(), DummyState(), telemetry, gsp=object(), memory=object())

        self.assertEqual(summary["total_count"], 15)
        self.assertGreaterEqual(summary["active_count"], 12)
        self.assertTrue(summary["ood"]["active"])
        self.assertIn("holoportation", summary["ood"]["reasons"])

        panel = format_epic_panel(summary)
        self.assertIn("OOD=", panel)
        self.assertIn("E42:on", panel)


if __name__ == "__main__":
    unittest.main()