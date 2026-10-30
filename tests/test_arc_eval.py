import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_eval import (
    calculate_confidence_interval, evaluate_tasks, extract_grid, load_tasks, save_report,
)


def task(task_id="example", outputs=([[1]],)):
    return {"id": task_id, "data": {"train": [{"input": [[0]], "output": [[1]]}],
                                  "test": [{"input": [[0]], "output": grid} for grid in outputs]}}


class TestARCProtocol(unittest.TestCase):
    def test_exact_grid_not_substring(self):
        self.assertEqual(extract_grid("Answer:\n[[1, 2],\n [3, 4]]"), "[[1,2],[3,4]]")
        for text in ("[[[1]]]", "[[1],[2,3]]", "[[10]]", "[[true]]", "[[]]",
                     "[[1]] then [[2", "[[1]] then [[x for x in range(2)]]"):
            with self.subTest(text=text):
                self.assertEqual(extract_grid(text), "")
        self.assertEqual(extract_grid("[[1]] then [[2]]"), "[[2]]")

    def test_every_test_pair_required(self):
        responses = iter(["[[1]]", "[[3]]"])
        report = evaluate_tasks([task(outputs=([[1]], [[2]]))], lambda *_: next(responses))
        self.assertEqual(report["summary"]["correct"], 0)
        self.assertEqual(report["summary"]["correct_pairs"], 1)
        self.assertEqual(report["summary"]["test_pairs"], 2)

    def test_errors_are_counted_and_invalidate_run(self):
        def fail(*_):
            raise RuntimeError("simulated backend failure")
        with self.assertLogs("arc_eval", level="ERROR"):
            report = evaluate_tasks([task()], fail, use_mfr=True)
        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["summary"]["generation_errors"], 1)
        self.assertEqual(report["summary"]["accuracy"], 0)
        self.assertFalse(report["summary"]["valid_for_comparison"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(RuntimeError):
                save_report(report, path, {})
            self.assertTrue(path.exists())
            with self.assertRaises(FileExistsError):
                save_report(report, path, {})

    def test_parse_failure_is_wrong_not_dropped(self):
        report = evaluate_tasks([task()], lambda *_: "No grid")
        self.assertEqual(report["summary"]["parse_failures"], 1)
        self.assertTrue(report["summary"]["valid_for_comparison"])
        self.assertEqual(report["summary"]["accuracy"], 0)

    def test_mfr_has_pair_denominator(self):
        responses = iter(["ENTITIES", "[[1]]", "No model", "[[2]]"])
        report = evaluate_tasks([task(outputs=([[1]], [[2]]))], lambda *_: next(responses), True)
        self.assertEqual(report["summary"]["correct"], 1)
        self.assertEqual(report["summary"]["mfr_compliance"], 0.5)

    def test_empty_and_duplicate_tasks_rejected(self):
        for tasks in ([], [task(), task()]):
            with self.assertRaises(ValueError):
                evaluate_tasks(tasks, lambda *_: "[[1]]")

    def test_seeded_selection_is_shared_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("b", "c", "a"):
                (Path(directory) / f"{name}.json").write_text(json.dumps(task()["data"]))
            first = load_tasks(directory, 2)
            self.assertEqual(first, load_tasks(directory, 2))
            self.assertEqual(len(first), 2)
            self.assertEqual(len(first[0]["sha256"]), 64)
            for limit in (0, -1):
                with self.assertRaises(ValueError):
                    load_tasks(directory, limit)

    def test_confidence_and_domain(self):
        lo95, hi95 = calculate_confidence_interval(1, 100)
        lo90, hi90 = calculate_confidence_interval(1, 100, 0.9)
        self.assertAlmostEqual(lo95, 0.00176743, places=7)
        self.assertAlmostEqual(hi95, 0.05448620, places=7)
        self.assertLess(lo95, lo90)
        self.assertGreater(hi95, hi90)
        for k, n in ((0, 0), (2, 1), (-1, 1)):
            with self.assertRaises(ValueError):
                calculate_confidence_interval(k, n)

    def test_runners_import_without_model_backends(self):
        import eval_arc
        import eval_arc_gguf
        self.assertIs(eval_arc.format_arc_prompt, eval_arc_gguf.format_arc_prompt)
