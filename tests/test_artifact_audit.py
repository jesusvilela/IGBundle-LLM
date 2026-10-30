from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.geometry_reference import disk_curvature, graph_curvature, render
from scripts.export_artifact_audit import fingerprint, reviewed_file


class TestArtifactAudit(unittest.TestCase):
    def test_changed_reviewed_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            original = b"reviewed source\n"
            path.write_bytes(original)
            self.assertEqual(reviewed_file(path, fingerprint(original)), original)
            path.write_bytes(b"changed source\n")
            with self.assertRaisesRegex(ValueError, "re-audit"):
                reviewed_file(path, fingerprint(original))

    def test_graph_curvature_signs(self):
        self.assertEqual(graph_curvature(0, 0), F(1, 4))
        self.assertEqual(graph_curvature(0, 0, saddle=True), F(-1, 4))
        for x in range(-4, 5):
            for y in range(-4, 5):
                self.assertGreater(graph_curvature(x, y), 0)
                self.assertLess(graph_curvature(x, y, saddle=True), 0)
                self.assertEqual(graph_curvature(x, y), -graph_curvature(x, y, saddle=True))
        self.assertNotEqual(graph_curvature(0, 0, saddle=True), graph_curvature(1, 0, saddle=True))

    def test_disk_curvature_exact(self):
        for x in range(-9, 10):
            for y in range(-9, 10):
                if x*x + y*y < 100:
                    self.assertEqual(disk_curvature(F(x, 10), F(y, 10)), -1)
        for point in ((1, 0), (1, 1)):
            with self.assertRaises(ValueError):
                disk_curvature(*point)

    def test_flat_cross_ratio_counterexample(self):
        def cross_ratio(a, b, c, d):
            return F(abs(a-c)*abs(b-d), abs(a-b)*abs(c-d))
        self.assertEqual(cross_ratio(0, 1, 2, 3), 4)
        self.assertEqual(cross_ratio(0, 2, 1, 3), F(1, 4))
        # Both sets lie on the same flat real line. The logarithm changes sign.

    def test_reference_svg_is_reproducible(self):
        expected = render()
        self.assertEqual((ROOT / "docs" / "assets" / "geometry-reference.svg").read_text(encoding="utf-8"), expected)
        ET.fromstring(expected)

    def test_published_directories_and_scope(self):
        data = ROOT / "docs" / "benchmark-results"
        for file, count, byte_count in (("gguf-dist-directory.json", 339, 5953216),
                                       ("gguf-arc-directory.json", 290, 5947510)):
            report = json.loads((data / file).read_text(encoding="utf-8"))
            self.assertEqual(report["tensor_count"], count)
            self.assertEqual(len(report["tensors"]), count)
            self.assertEqual(report["prefix_bytes"], byte_count)
            self.assertEqual(report["tensor_payload_bytes_read"], 0)
            self.assertEqual(report["qwen2_standard_names"]["extra"], [])
            self.assertEqual(report["qwen2_standard_names"]["missing"], [])
            self.assertEqual(report["marker_tensor_names"], [])
        audit = json.loads((data / "artifact-audit.json").read_text(encoding="utf-8"))
        verification = json.loads((data / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(len(audit["images"]), 9)
        for source in audit["sources"]:
            for excerpt in source["excerpts"]:
                self.assertEqual(len(excerpt["lines"]), excerpt["end"] - excerpt["start"] + 1)
        for image in audit["images"][:3]:
            # Published copies must not remain the original unlabeled illustrations.
            path = ROOT / "thesis" / "figures" / Path(image["path"]).name
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            self.assertNotEqual(digest, image["sha256"])
            self.assertEqual(struct.unpack(">II", content[16:24]),
                             (image["width"], image["height"] + round(image["width"] * 0.075)))
            self.assertEqual(digest, verification["publication_asset_sha256"][path.relative_to(ROOT).as_posix()])


if __name__ == "__main__":
    unittest.main()
