import ast
import io
import math
import os
from pathlib import Path
import sys
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from igbundle.demo_evidence import DemoTelemetry, format_metrics


def observations(tokens=1, sigma=1.0, p=(0.5, 0.5)):
    return np.full((1, tokens, 1, 1), sigma), np.tile(p, (1, tokens, 1, 1))


class TelemetryTests(unittest.TestCase):
    def test_missing_is_unavailable_not_zero(self):
        telemetry = DemoTelemetry()
        self.assertIsNone(telemetry.summary())
        self.assertIsNone(telemetry.mean_probabilities())
        self.assertIn("unavailable", format_metrics(None))
        self.assertNotIn("0.0", format_metrics(None))

    def test_entropy_and_arbitrary_interval_have_exact_meaning(self):
        telemetry = DemoTelemetry()
        sigma, p = observations(4)
        sigma[0, :, 0, 0] = [.5, 2., .75, 1.]
        telemetry.add(sigma, p)
        result = telemetry.summary()
        self.assertEqual(result["sigma_interval_percent"], 50.)
        self.assertAlmostEqual(result["mean_sigma"], 1.0625)
        self.assertAlmostEqual(result["mean_categorical_entropy"], math.log(2))
        text = format_metrics(result)
        self.assertIn("not curvature", text)
        self.assertIn("arbitrary interval", text)
        self.assertNotIn("MFR", text)

    def test_variable_decode_lengths_are_observation_weighted(self):
        telemetry = DemoTelemetry()
        telemetry.add(*observations(3, sigma=2.0))
        telemetry.add(*observations(1, sigma=6.0, p=(1., 0.)))
        result = telemetry.summary()
        self.assertEqual(result["mean_sigma"], 3.0)
        self.assertAlmostEqual(result["mean_categorical_entropy"], .75 * math.log(2))
        np.testing.assert_allclose(telemetry.mean_probabilities(), [[.625, .375]])
        self.assertEqual(result["categorical_rows"], 4)

    def test_low_precision_rounding_is_normalized_with_explicit_tolerance(self):
        telemetry = DemoTelemetry()
        telemetry.add(*observations(p=(.499, .499)))
        self.assertAlmostEqual(telemetry.summary()["mean_categorical_entropy"], math.log(2))
        with self.assertRaises(ValueError):
            telemetry.add(*observations(p=(.4, .4)))

    def test_malformed_observations_do_not_change_accumulators(self):
        telemetry = DemoTelemetry()
        telemetry.add(*observations())
        before = telemetry.summary()
        bad = [
            observations(sigma=float("nan")), observations(sigma=float("inf")),
            observations(sigma=0.), observations(sigma=-1.),
            observations(p=(-.1, 1.1)), observations(p=(float("nan"), .5)),
            (np.ones(2), np.ones(2)), observations(0),
            (np.ones((1, 1, 2, 1)), np.full((1, 1, 2, 2), .5)),
        ]
        for sigma, p in bad:
            with self.subTest(shape=sigma.shape), self.assertRaises(ValueError):
                telemetry.add(sigma, p)
            self.assertEqual(before, telemetry.summary())

    def test_retained_samples_are_bounded_and_reset_clears_all(self):
        telemetry = DemoTelemetry()
        telemetry.add(*observations(10000))
        telemetry.add(*observations(10000, sigma=3.0))
        self.assertEqual(telemetry.sigma_samples.size, 4096)
        self.assertEqual(telemetry.summary()["sigma_count"], 20000)
        self.assertEqual(telemetry.summary()["mean_sigma"], 2.0)
        telemetry.reset()
        self.assertIsNone(telemetry.summary())
        self.assertIsNone(telemetry.probability_sum)
        self.assertEqual(telemetry.sigma_samples.size, 0)

    def test_accumulator_overflow_is_explicit_and_does_not_mutate(self):
        telemetry = DemoTelemetry()
        with self.assertRaises((ValueError, FloatingPointError)):
            telemetry.add(*observations(2, sigma=1.7e308))
        self.assertIsNone(telemetry.summary())


class DemoSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        selected = {
            "_figure_image", "_unavailable_plot", "plot_dispersion", "plot_categorical",
            "plot_training_metrics", "plot_ablation_comparison", "load_topo_stats",
            "generate_with_viz",
            "launch_app",
        }
        nodes = [node for node in self.tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name in selected)
                 or (isinstance(node, ast.ClassDef) and node.name == "HookManager")]
        self.telemetry = DemoTelemetry()
        self.ns = dict(
            io=io, plt=plt, Image=Image, TELEMETRY=self.telemetry,
            GENERATION_LOCK=threading.Lock(), format_metrics=format_metrics,
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), self.ns)

    def tearDown(self):
        self.assertEqual(plt.get_fignums(), [], "Plot leaked an open figure")

    def test_rendered_placeholders_have_no_synthetic_data(self):
        saved = []
        from matplotlib.figure import Figure
        original_savefig = Figure.savefig

        def capture(fig, *args, **kwargs):
            saved.append(fig)
            return original_savefig(fig, *args, **kwargs)

        with patch.object(Figure, "savefig", capture):
            for name in ("plot_training_metrics", "plot_ablation_comparison"):
                image = self.ns[name]()
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)
                image.close()
        for fig in saved:
            ax = fig.axes[0]
            self.assertEqual(len(ax.lines), 0)
            self.assertEqual(len(ax.patches), 0)
            self.assertIn("unavailable", ax.get_title())

    def test_statistics_do_not_read_unbound_local_claims(self):
        with patch("builtins.open", side_effect=AssertionError("Unexpected local claims")):
            rows = self.ns["load_topo_stats"]()
        self.assertTrue(all(len(row) == 2 for row in rows))
        self.assertNotIn("131.5", repr(rows))
        self.assertNotIn("94.2", repr(rows))

    def test_observed_plots_render_and_no_hooks_remains_unavailable(self):
        self.ns["generate_response"] = lambda message, history: "reply"
        response, first, second, text = self.ns["generate_with_viz"]("test", [])
        self.assertEqual(response, "reply")
        self.assertIn("unavailable", text)
        first.close()
        second.close()
        self.telemetry.add(*observations(3))
        for name in ("plot_dispersion", "plot_categorical"):
            image = self.ns[name]()
            self.assertEqual(image.format, None)  # Independent copy; buffer is closed.
            image.close()

    def test_request_reset_prevents_previous_telemetry_leak(self):
        self.telemetry.add(*observations())
        self.ns["generate_response"] = lambda message, history: "reply"
        _, first, second, text = self.ns["generate_with_viz"]("next", [])
        self.assertIsNone(self.telemetry.summary())
        self.assertIn("unavailable", text)
        first.close()
        second.close()

    def test_removed_claims_and_mandatory_optional_imports_do_not_return(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        for claim in ("131.5", "94.2", "30% faster", "p < 0.001", "Optimal Range"):
            self.assertNotIn(claim, source)
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, ("unsloth", "generate_braintop_viz", "seaborn"))

    def test_hook_capture_uses_sigma_and_p_and_converts_before_numpy(self):
        sigma, p = observations()
        sigma_tensor, p_tensor = Mock(), Mock()
        sigma_tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = sigma
        p_tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = p
        manager = self.ns["HookManager"](Mock())
        manager._capture(None, None, (None, SimpleNamespace(sigma=sigma_tensor, p=p_tensor)))
        self.assertEqual(self.telemetry.summary()["sigma_count"], 1)
        sigma_tensor.detach.return_value.float.assert_called_once()
        p_tensor.detach.return_value.float.assert_called_once()

    def test_hook_replacement_and_cleanup_do_not_accumulate_handles(self):
        class FakeAdapter:
            def register_forward_hook(self, hook):
                return Mock()

        instance = FakeAdapter()
        manager = self.ns["HookManager"](SimpleNamespace(modules=lambda: [instance]))
        with patch.dict(sys.modules, {
            "igbundle.modules.adapter": SimpleNamespace(IGBundleAdapter=FakeAdapter)
        }):
            manager.attach()
            first = manager.hooks[0]
            manager.attach()
            first.remove.assert_called_once()
            self.assertEqual(len(manager.hooks), 1)
            second = manager.hooks[0]
            manager.detach()
            manager.detach()
            second.remove.assert_called_once()
            with self.assertRaises(ValueError):
                self.ns["HookManager"](SimpleNamespace(modules=lambda: [])).attach()

    def test_real_gradio_ui_constructs_with_stubbed_model_and_no_server(self):
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        import gradio as gr

        self.ns["gr"] = gr
        self.ns["EVIDENCE_URL"] = "https://jesusvilela.github.io/IGBundle-LLM/wiki.html"
        self.ns["EXAMPLE_TASKS"] = {"Example": ["test"]}
        self.ns["load_model"] = Mock(return_value=(object(), object()))
        hooks = Mock()
        self.ns["HookManager"] = Mock(return_value=hooks)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"GRADIO_TEMP_DIR": directory}):
                with patch.object(gr.Blocks, "launch", autospec=True) as launch:
                    self.ns["launch_app"]("config", "checkpoint")
                    launch.assert_called_once()
                    demo = launch.call_args.args[0]
                    try:
                        tabs = [component["props"]["label"]
                                for component in demo.config["components"]
                                if component["type"] == "tabitem"]
                        self.assertEqual(tabs, ["Inference", "Evidence availability", "Interpretation"])
                        self.assertNotIn("131.5", repr(demo.config))
                    finally:
                        demo.close()
        hooks.attach.assert_called_once()
        hooks.detach.assert_called_once()


if __name__ == "__main__":
    unittest.main()
