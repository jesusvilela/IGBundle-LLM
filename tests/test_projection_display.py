import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_braintop_viz as display


class ProjectionDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "adapter_weights.pt"
        self.output = self.root / "display.html"

    def save(self, state=None):
        if state is None:
            state = {"input_proj.weight": torch.tensor([[0., 1.], [-1., .5]])}
        torch.save(state, self.weights)

    def test_missing_checkpoint_never_generates_scaffold(self):
        with patch.object(torch, "load") as load:
            with self.assertRaises(FileNotFoundError):
                display.generate_viz(self.root, self.output)
            load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_read_is_tensor_only_cpu_and_hash_binds_exact_file(self):
        self.save()
        with patch.object(torch, "load", wraps=torch.load) as load:
            value, key, digest = display.read_projection(self.root)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(key, "input_proj.weight")
        self.assertEqual(value.shape, (2, 2))
        self.assertEqual(digest, hashlib.sha256(self.weights.read_bytes()).hexdigest())

    def test_rejected_load_is_not_retried(self):
        self.save()
        with patch.object(torch, "load", side_effect=ValueError("Rejected pickle")) as load:
            with self.assertRaisesRegex(ValueError, "Rejected pickle"):
                display.generate_viz(self.root, self.output)
        self.assertEqual(load.call_count, 1)
        self.assertFalse(self.output.exists())

    def test_missing_ambiguous_and_inexact_keys_fail(self):
        for state in ({}, {"xinput_proj.weight": torch.ones(1, 1)},
                      {"a.input_proj.weight": torch.ones(1, 1),
                       "b.input_proj.weight": torch.ones(1, 1)}):
            with self.subTest(keys=list(state)):
                self.save(state)
                with self.assertRaises(ValueError):
                    display.read_projection(self.root)
        _, key, _ = display.read_projection(self.root, "b.input_proj.weight")
        self.assertEqual(key, "b.input_proj.weight")
        with self.assertRaises(ValueError):
            display.read_projection(self.root, "unknown")

    def test_invalid_selected_tensors_preserve_previous_output(self):
        self.output.write_text("previous output", encoding="utf-8")
        invalid = [
            torch.empty(0, 2), torch.ones(2), torch.ones(1, 1, dtype=torch.int64),
            torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
            torch.ones(1, 1, dtype=torch.complex64), torch.ones(1, 1).to_sparse(),
            torch.empty(1, 1, device="meta"), "not a tensor",
        ]
        for value in invalid:
            with self.subTest(value=repr(value)):
                self.save({"input_proj.weight": value})
                with self.assertRaises(ValueError):
                    display.generate_viz(self.root, self.output)
                self.assertEqual(self.output.read_text(encoding="utf-8"), "previous output")

    def test_size_limits_reject_without_rendering(self):
        self.save()
        with patch.object(display, "MAX_CHECKPOINT_BYTES", 1), patch.object(torch, "load") as load:
            with self.assertRaises(ValueError):
                display.read_projection(self.root)
            load.assert_not_called()
        with patch.object(display, "MAX_PROJECTION_VALUES", 1):
            with self.assertRaises(ValueError):
                display.read_projection(self.root)
        self.assertFalse(self.output.exists())

    def test_filenames_do_not_select_geometry_and_bytes_are_deterministic(self):
        self.save()
        first = self.root / "hyperbolic.html"
        second = self.root / "euclidean.html"
        display.generate_viz(self.root, first)
        display.generate_viz(self.root, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        text = first.read_text(encoding="utf-8")
        self.assertIn(display.NOTICE, text)
        svg = ET.fromstring(text[text.index("<svg "):text.index("</svg>") + 6])
        ns = {"s": "http://www.w3.org/2000/svg"}
        cells = svg.findall(".//s:g/s:rect", ns)
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0].attrib["fill"], "rgb(255,255,255)")
        self.assertEqual(cells[1].attrib["fill"], "rgb(255,0,0)")
        self.assertEqual(cells[2].attrib["fill"], "rgb(0,0,255)")
        self.assertIn("row 1, column 1: 0.5", text)

    def test_lite_prefix_is_bounded_and_zero_bfloat16_is_supported(self):
        self.save({"input_proj.weight": torch.zeros(65, 66, dtype=torch.bfloat16)})
        for lite, count in ((True, 32 * 32), (False, 64 * 64)):
            display.generate_viz(self.root, self.output, lite)
            text = self.output.read_text(encoding="utf-8")
            self.assertEqual(text.count("<title>row "), count)
            self.assertIn("maximum absolute value 0.", text)

    def test_unbound_metadata_and_wrong_output_extension_fail(self):
        with self.assertRaises(ValueError):
            display.generate_viz(self.root, self.output, node_metadata={})
        with self.assertRaises(ValueError):
            display.generate_viz(self.root, self.root / "output.png")
        self.assertFalse(self.output.exists())

    def test_tensor_key_is_escaped_in_html(self):
        self.save({"<script>.input_proj.weight": torch.ones(1, 1)})
        display.generate_viz(self.root, self.output)
        text = self.output.read_text(encoding="utf-8")
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;.input_proj.weight", text)


if __name__ == "__main__":
    unittest.main()
