"""Backend-independent ARC-AGI subset protocol; no model imports or downloads."""

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from statistics import NormalDist

from mfr_utils import construct_phase1_prompt, construct_phase2_prompt

PROTOCOL = "arc-all-test-pairs-exact-v1"
LOGGER = logging.getLogger(__name__)


def validate_grid(grid):
    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
        raise ValueError("An ARC grid must contain 1 to 30 rows")
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    if not 1 <= width <= 30:
        raise ValueError("An ARC grid must contain 1 to 30 columns")
    if any(not isinstance(row, list) or len(row) != width for row in grid):
        raise ValueError("An ARC grid must be rectangular")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("ARC colors must be integers from 0 to 9")
    return grid


def format_grid(grid):
    return json.dumps(validate_grid(grid), separators=(",", ":"))


def extract_grid(text):
    """Parse only the last grid-shaped top-level array, never a target substring.

    An invalid final candidate is a parse failure even if an earlier array was
    valid. This is a declared parser convention, not a semantic answer judge.
    """
    depth = 0
    start = None
    candidate = None
    for index, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0:
                block = text[start:index + 1]
                if block[1:].lstrip().startswith("["):
                    candidate = block
    if depth and text[start + 1:].lstrip().startswith("["):
        return ""
    if candidate is None:
        return ""
    try:
        return format_grid(json.loads(candidate))
    except (ValueError, TypeError):
        return ""


def format_arc_prompt(example, num_shots=1, test_index=0):
    prompt = "The following logical reasoning puzzle involves transforming input grids to output grids.\n\n"
    for index, pair in enumerate(example["train"][:num_shots]):
        prompt += (f"Example {index + 1}:\nInput: {format_grid(pair['input'])}\n"
                   f"Output: {format_grid(pair['output'])}\n\n")
    pair = example["test"][test_index]
    prompt += f"Test:\nInput: {format_grid(pair['input'])}\nOutput:"
    return prompt, validate_grid(pair["output"])


def calculate_confidence_interval(k, n, confidence=0.95):
    """Wilson interval for nonempty binomial observations."""
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n == 0:
        raise ValueError("Wilson interval requires integers 0 <= k <= n and n > 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    proportion = k / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_tasks(data_path, limit=None, seed=42):
    """Sort before seeded selection; validate data before loading any model."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    files = sorted(Path(data_path).glob("*.json"))
    if not files:
        raise ValueError(f"No ARC task JSON files found in {data_path}")
    random.Random(seed).shuffle(files)
    tasks = []
    for path in files[:limit]:
        raw = path.read_bytes()
        data = json.loads(raw)
        for split in ("train", "test"):
            if not isinstance(data.get(split), list) or not data[split]:
                raise ValueError(f"{path.name}: missing or empty {split} pairs")
            for pair in data[split]:
                validate_grid(pair["input"])
                validate_grid(pair["output"])
        tasks.append({"id": path.stem, "data": data, "sha256": hashlib.sha256(raw).hexdigest()})
    return tasks


def evaluate_tasks(tasks, generate, use_mfr=False):
    """Score all test pairs; a task is solved only when every pair is correct.

    Backend/data failures remain in the denominator and invalidate the run.
    Parse failures are incorrect answers. Callers must reject invalid reports.
    """
    if not tasks or len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Evaluation requires nonempty, uniquely identified tasks")
    details = []
    for task in tasks:
        predictions = []
        if not task["data"].get("test"):
            raise ValueError(f"{task['id']}: no test pairs")
        for test_index in range(len(task["data"]["test"])):
            prompt, target = format_arc_prompt(task["data"], test_index=test_index)
            entry = {"test_index": test_index, "target": format_grid(target),
                     "correct": False, "mfr_compliant": False}
            try:
                if use_mfr:
                    model_text = generate(construct_phase1_prompt(prompt), 1024)
                    entry["model_text"] = model_text
                    entry["mfr_compliant"] = "## PROBLEM MODEL" in model_text or "ENTITIES" in model_text
                    prompt = construct_phase2_prompt(prompt, model_text)
                generated = generate(prompt, 2048)
                parsed = extract_grid(generated)
                entry.update(generated=generated, extracted=parsed,
                             parse_failure=not bool(parsed), correct=parsed == entry["target"])
            except (RuntimeError, OSError, ValueError, TypeError, KeyError) as error:
                LOGGER.exception("ARC generation failed for %s test %d", task["id"], test_index)
                entry["error"] = f"{type(error).__name__}: {error}"
            predictions.append(entry)
        details.append({"task_id": task["id"], "source_sha256": task.get("sha256"),
                        "correct": all(pair["correct"] for pair in predictions),
                        "predictions": predictions})
    pairs = [pair for task in details for pair in task["predictions"]]
    correct = sum(task["correct"] for task in details)
    errors = sum("error" in pair for pair in pairs)
    return {
        "protocol": PROTOCOL,
        "summary": {
            "total": len(tasks), "correct": correct, "accuracy": correct / len(tasks),
            "confidence_interval": calculate_confidence_interval(correct, len(tasks)),
            "test_pairs": len(pairs), "correct_pairs": sum(pair["correct"] for pair in pairs),
            "parse_failures": sum(pair.get("parse_failure", False) for pair in pairs),
            "generation_errors": errors, "valid_for_comparison": errors == 0,
            "mfr_compliance": sum(pair["mfr_compliant"] for pair in pairs) / len(pairs) if use_mfr else None,
            "mfr_denominator": "test_pairs",
        },
        "details": details,
    }


def save_report(report, output_path, metadata):
    report["metadata"] = metadata
    # Do not silently overwrite an earlier experiment.
    with Path(output_path).open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    if not report["summary"]["valid_for_comparison"]:
        raise RuntimeError(f"Run contains generation errors; invalid report saved to {output_path}")
