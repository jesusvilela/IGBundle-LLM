import csv
import json
import math
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

    def test_recorded_companion_metadata(self):
        directory = ROOT / "docs" / "benchmark-results"
        record = json.loads((directory / "companion-state-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(record["protocol"], "companion-state-audit-v1")
        reports = {report["file_name"]: report for report in record["checkpoints"]}
        self.assertEqual(len(reports), 4)
        self.assertEqual(record["total_tensor_descriptors"], 226)
        self.assertEqual(record["total_metadata_read_bytes"], 52399)
        self.assertEqual(sum(report["metadata_read_bytes"] for report in reports.values()), 52399)
        for filename, count in (("adapter_refined.pt", 81), ("arc_adapter.pt", 73),
                                ("arc_projector.pt", 11), ("adapter_weights.pt", 61)):
            report = reports[filename]
            self.assertEqual(report["protocol"], "torch-state-metadata-v1")
            self.assertEqual(report["tensor_count"], count)
            self.assertEqual(len(report["tensors"]), count)
            self.assertEqual(report["storage_count"], count)
            stores = {storage["key"]: storage for storage in report["storages"]}
            self.assertEqual(len(stores), count)
            self.assertEqual(report["tensor_storage_members_opened"], 0)
            self.assertLessEqual(report["metadata_bytes"], 1024 * 1024)
            self.assertLessEqual(report["zip_directory_bytes"], 1024 * 1024)
            self.assertEqual(report["sum_tensor_elements"], sum(math.prod(t["shape"]) for t in report["tensors"]))
            for key in ("metadata_sha256", "zip_directory_sha256"):
                self.assertRegex(report[key], r"^[0-9a-f]{64}$")
            for tensor in report["tensors"]:
                storage = stores[tensor["storage_key"]]
                self.assertEqual(tensor["dtype"], storage["dtype"])
                itemsize = {"float32": 4, "bfloat16": 2}[storage["dtype"]]
                self.assertEqual(storage["declared_bytes"], storage["elements"] * itemsize)
                self.assertEqual(len(tensor["shape"]), len(tensor["stride"]))
                span = 1 + sum((size - 1) * step for size, step in zip(tensor["shape"], tensor["stride"]))
                self.assertLessEqual(tensor["storage_offset"] + span, storage["elements"])
        arc = {tensor["name"]: tensor for tensor in reports["arc_adapter.pt"]["tensors"]}
        gguf = json.loads((directory / "gguf-arc-directory.json").read_text(encoding="utf-8"))
        self.assertEqual(arc["input_proj.weight"]["shape"], [896, 3584])
        self.assertEqual(arc["output_proj.weight"]["shape"], [3584, 40])
        self.assertEqual(gguf["metadata"]["qwen2.embedding_length"], 896)
        self.assertNotEqual(arc["input_proj.weight"]["shape"][1], gguf["metadata"]["qwen2.embedding_length"])
        for tensor in reports["arc_projector.pt"]["tensors"]:
            embedded = arc["vision_projector." + tensor["name"]]
            self.assertEqual(tensor["shape"], embedded["shape"])
            self.assertEqual(tensor["dtype"], "bfloat16")
            self.assertEqual(embedded["dtype"], "float32")
        dist = reports["adapter_refined.pt"]["tensors"]
        self.assertEqual(sum(t["name"].startswith(("vision_projector.", "vision_attn.", "vision_norm."))
                             for t in dist), 17)
        self.assertEqual(sum(t["name"].startswith("geodesic_attn.") for t in dist), 5)

    def test_native_gguf_recorded_witness(self):
        path = ROOT / "docs" / "benchmark-results" / "native-gguf-reference.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["protocol"], "native-qwen2-f32-v1")
        cases = record["comparisons"]
        self.assertEqual({(case["tied_output"], len(case["tokens"])) for case in cases},
                         {(False, 4), (False, 6), (True, 4), (True, 6)})
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(case["logit_shape"], [len(case["tokens"]), 259])
            self.assertEqual(len(case["argmax_ids"]), len(case["tokens"]))
            self.assertLess(case["parameters"], record["acceptance"]["parameters_below"])
            self.assertLess(case["file_bytes"], record["acceptance"]["fixture_bytes_below"])
            for key in ("file_sha256", "directory_prefix_sha256"):
                self.assertRegex(case[key], r"^[0-9a-f]{64}$")
            for key in ("full_max_abs_error", "incremental_max_abs_error", "batch_max_abs_error"):
                self.assertTrue(math.isfinite(case[key]) and 0 <= case[key] < record["acceptance"]["atol"])
        counterexample = record["payload_counterexample"]
        self.assertIs(counterexample["same_directory_report_except_filename"], True)
        self.assertEqual(counterexample["original_file_sha256"], cases[0]["file_sha256"])
        self.assertEqual(counterexample["directory_prefix_sha256"], cases[0]["directory_prefix_sha256"])
        self.assertNotEqual(counterexample["original_file_sha256"], counterexample["modified_file_sha256"])
        self.assertGreater(counterexample["max_abs_logit_change"],
                           record["acceptance"]["payload_logit_change_above"])
        self.assertEqual(counterexample["negation_max_abs_error"], 0)
        self.assertEqual(len(record["malformed_shapes"]), 2)
        for malformed in record["malformed_shapes"]:
            self.assertEqual(malformed["inspector_status"], "failed")
            self.assertEqual(malformed["native_exit_code"], 2)

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
