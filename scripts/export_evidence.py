"""Export prompt-free historical measurements from an explicitly supplied archive.

This copies recorded grades, not their correctness as scientific measurements.
It never loads model weights or publishes prompts, targets, or generations.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc_eval import extract_grid

VARIANTS = {
    "merged_cp300": "ig-bundlellmv2-hamiltonians",
    "base_qwen": "qwen2-5-7b-instruct",
    "reconstructed_adapter_composite": "igbundle-cp300-corrected",
}
BENCHMARKS = ("mmlupro", "arc", "math", "gsm8k", "ifbench", "probes")


def export(archive, research_tree, cp300_index, output):
    source_hashes = {}

    def read(path, relative_name):
        raw = path.read_bytes()
        source_hashes[relative_name] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8-sig")

    paired = []
    for benchmark in BENCHMARKS:
        arms = {}
        for variant, directory in VARIANTS.items():
            relative = f"aa-index-2026-10-27/{directory}/{benchmark}.jsonl"
            rows = [json.loads(line) for line in read(
                archive / "aa-index-2026-10-27" / directory / f"{benchmark}.jsonl", relative
            ).splitlines() if line.strip()]
            if not rows or len({row["id"] for row in rows}) != len(rows):
                raise ValueError(f"{relative}: empty or duplicate item IDs")
            if any(type(row["ok"]) is not bool for row in rows):
                raise ValueError(f"{relative}: non-boolean recorded grade")
            arms[variant] = {row["id"]: row["ok"] for row in rows}
        ids = set(arms["merged_cp300"])
        if any(set(arm) != ids for arm in arms.values()):
            raise ValueError(f"{benchmark}: unpaired item IDs")
        for item_id in sorted(ids):
            paired.append({"benchmark": benchmark, "item_id": item_id,
                           **{name: int(arm[item_id]) for name, arm in arms.items()}})

    nll = json.loads(read(archive / "falsification-matched-2026-09-20" / "results.json",
                         "falsification-matched-2026-09-20/results.json"))
    arc = json.loads(read(research_tree / "arc_evaluation_results.json", "arc_evaluation_results.json"))
    audit = json.loads(read(research_tree / "experiments" / "falsification" / "results" /
                           "A_faithfulness_audit.json", "A_faithfulness_audit.json"))
    audit_source = read(research_tree / "experiments" / "falsification" / "audit_faithfulness.py",
                        "audit_faithfulness.py")
    tensor_index = json.loads(read(cp300_index, "cp300/model.safetensors.index.json"))
    tensor_names = list(tensor_index["weight_map"])
    custom_names = [name for name in tensor_names
                    if any(marker in name.lower() for marker in ("adapter", "igbundle", "geometric", "metric_chol"))]
    arc_rows = []
    for row in arc["details"]:
        generated = row.get("generated", "")
        target = row.get("target", "")
        parsed = extract_grid(generated)
        arc_rows.append({
            "task_id": row["task_id"],
            "recorded_correct": bool(row.get("correct")),
            "strict_first_pair_correct": bool(target and parsed == target),
            "generation_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "recorded_error": "error" in row,
        })
    provenance = {
        "scope": "Historical recorded measurements; no model inference rerun or independent regrading.",
        "source_sha256": source_hashes,
        "october_checkpoint_index": {
            "tensor_count": len(tensor_names),
            "custom_adapter_key_matches": custom_names,
            "scope": "Inspection of the surviving cp300 safetensors index only; not a GGUF tensor audit or proof of the served model identity.",
        },
        "september": {
            "historical_source_commit_local_only": nll["historical_source_commit"],
            "geometric_checkpoint_sha256": nll["geometric_checkpoint_sha256"],
            "control_checkpoint_sha256": nll["control_checkpoint_sha256"],
            "matching": nll["matching"],
            "environment_reported_by_run": nll["environment"],
            "public_reproduction_limit": "Original dataset revision and complete loader are not bundled.",
        },
        "operator_audit": {
            "inspected_source_snapshot": "historical-audit-source.py.txt",
            "timestamp_reported_by_run": audit["eval_timestamp"],
            "reported_pass_count": audit["tests_passed"],
            "source_snapshot_lines": len(audit_source.splitlines()),
            "immutable_run_source_and_checkpoint_binding": False,
            "limitations": [
                "Inspected loader accepts missing weights (random initialization) and partial state loads.",
                "Transport check measures coordinate displacement, not metric compatibility or holonomy.",
                "Triangle check clamps scalar norms without projecting points.",
                "Energy check uses squared coordinate norm, not the Hamiltonian; drift is a fraction despite the pct key.",
                "Entropy and divergence from uniform do not establish specialization or fiber diversity.",
                "These observations do not identify which loader path produced the historical record.",
            ],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "historical-audit-source.py.txt").write_text(
        audit_source.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    with (output / "aa-paired-outcomes-2026-10-27.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark", "item_id", *VARIANTS], lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    for name, data in (("provenance.json", provenance), ("arc-agi-rescore.json", arc_rows)):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8", newline="\n")
    print(f"Exported {len(paired)} paired outcome rows and {len(arc_rows)} historical first-pair rescoring rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--research-tree", required=True, type=Path)
    parser.add_argument("--cp300-index", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "benchmark-results")
    args = parser.parse_args()
    export(args.archive, args.research_tree, args.cp300_index, args.output)
