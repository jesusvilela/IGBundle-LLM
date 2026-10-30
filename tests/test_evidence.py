import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_evidence import check_site_links, exact_mcnemar, verify


class TestEvidence(unittest.TestCase):
    def test_published_arithmetic_and_links(self):
        report = verify(ROOT)
        self.assertEqual(report["september"]["control_wins"], 24)
        self.assertAlmostEqual(report["october"]["merged_cp300"]["weighted_index"], 60.0, delta=0.05)
        saved = json.loads((ROOT / "docs" / "benchmark-results" / "verification.json").read_text(encoding="utf-8"))
        self.assertEqual(report, saved, "Regenerate verification.json after changing evidence or site links")

    def test_exact_paired_test(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(2, 0), 0.5)
        self.assertEqual(exact_mcnemar(3, 2), 1.0)

    def test_tampered_pair_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "docs", root / "docs")
            path = root / "docs" / "benchmark-results" / "aa-paired-outcomes-2026-10-27.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = reader.fieldnames
            rows[0]["merged_cp300"] = str(1 - int(rows[0]["merged_cp300"]))
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "accuracy mismatch"):
                verify(root)

    def test_missing_static_target_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "index.html").write_text('<a href="missing.html">missing</a>')
            with self.assertRaisesRegex(ValueError, "Missing static-site target"):
                check_site_links(root)
