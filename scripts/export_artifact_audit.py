"""Export selected source/image fingerprints from the research archive, without weights."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "igbundle-llm/generate_readme_visualizations.py": [(195, 211), (240, 294), (313, 320), (390, 423)],
    "igbundle-llm/thesis_figures.py": [(91, 174)],
    "igbundle-llm/generate_visual_report.py": [(10, 16), (39, 68)],
    "generate_benchmark_report.py": [(24, 80)],
    "igbundle-llm/generate_braintop_viz.py": [(23, 58), (93, 102), (161, 188)],
    "igbundle-llm/dist/run_inference_integrated.py": [(11, 34), (72, 91)],
    "igbundle-llm/export_refined_package.py": [(10, 17), (37, 43), (75, 86)],
    "igbundle_arc_deployment/README.md": [(1, 21)],
    "llmos/research/neural_info_geometry/info_geometry_experiments.py": [(204, 228), (268, 289)],
}
IMAGES = [
    ("igbundle-llm/thesis/figures/figure_4_dynamics.png", "synthetic",
     "Loss and sigma are exponentials plus random noise, not recorded training telemetry."),
    ("igbundle-llm/thesis/figures/figure_5_affinity.png", "synthetic",
     "Block affinities are assigned random values, not extracted from a checkpoint."),
    ("igbundle-llm/thesis/figures/figure_7_svd.png", "synthetic",
     "The two spectra are explicit power laws with an imposed cutoff, not an SVD."),
    ("igbundle-llm/output/readme_visuals/riemannian_geometry.png", "mathematically_mislabeled",
     "The drawn downward paraboloid has positive induced Gaussian curvature. "
     "The negative heatmap and transport rotations are assigned independently."),
    ("igbundle-llm/output/readme_visuals/information_geometry.png", "synthetic",
     "Random A A^T, hand-specified paths and exponential curves do not measure Fisher geometry or speedup."),
    ("igbundle-llm/output/readme_visuals/before_after_comparison.png", "schematic_not_validation",
     "The 'corrected mathematical rigor' side is an illustration, not verification of the listed laws."),
    ("igbundle-llm/benchmark_graphical_report.png", "unmatched_comparison",
     "Generator compares AIME to MATH, labels ARC-Challenge as ARC-AGI, and has heuristic fallbacks."),
    ("igbundle-llm/output/igbundle_topology_riemannian_final.png", "checkpoint_binding_unresolved",
     "A graph layout does not determine intrinsic curvature. Inspected renderer imposes curvature=-1 "
     "and permits random embeddings when weights/projection are missing; not proof this image used that fallback."),
    ("llmos/research/neural_info_geometry/experiments/neural_info_geometry_results.png",
     "toy_experiment_not_llm_evidence",
     "Associated source uses a small MLP and synthetic data; its log cross-ratio is not sectional curvature."),
]
ALGEBRA_TERMS = re.compile(
    r"hypercomplex|quaternion|octonion|sedenion|cayley|clifford|complex64|complex128|torch\.complex",
    re.IGNORECASE,
)


def small_file(path):
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"Refusing non-small evidence file: {path.name}")
    return path.read_bytes()


def fingerprint(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def reviewed_file(path, expected):
    data = small_file(path)
    if fingerprint(data) != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
        raise ValueError(f"Reviewed artifact changed; re-audit before reclassifying: {path.name}")
    return data


def export(archive):
    archive = Path(archive)
    reference = json.loads((ROOT / "docs" / "benchmark-results" / "artifact-audit.json").read_text(encoding="utf-8"))
    expected_sources = {row["path"]: row for row in reference["sources"]}
    expected_images = {row["path"]: row for row in reference["images"]}
    sources = []
    for relative, ranges in SOURCES.items():
        data = reviewed_file(archive / relative, expected_sources[relative])
        lines = data.decode("utf-8-sig").splitlines()
        if any(start < 1 or end > len(lines) for start, end in ranges):
            raise ValueError(f"Source excerpt range no longer fits: {relative}")
        sources.append({
            "path": relative, **fingerprint(data),
            "excerpts": [{"start": start, "end": end, "lines": lines[start - 1:end]}
                         for start, end in ranges],
        })
    images = []
    for relative, classification, finding in IMAGES:
        data = reviewed_file(archive / relative, expected_images[relative])
        if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError(f"Not a PNG: {relative}")
        width, height = struct.unpack(">II", data[16:24])
        images.append({"path": relative, **fingerprint(data), "width": width, "height": height,
                       "classification": classification, "finding": finding})
    scans = []
    for relative in ("igbundle-llm/src/igbundle", "igbundle-llm/dist/igbundle_lib/igbundle"):
        files, matches = [], []
        for path in sorted((archive / relative).rglob("*.py")):
            data = small_file(path)
            name = path.relative_to(archive).as_posix()
            files.append({"path": name, **fingerprint(data)})
            for number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
                if ALGEBRA_TERMS.search(line):
                    matches.append({"path": name, "line": number})
        if not files:
            raise ValueError(f"Empty source scan: {relative}")
        scans.append({"root": relative, "pattern": ALGEBRA_TERMS.pattern,
                      "files": files, "matches": matches})
    return {
        "protocol": "archive-artifact-audit-v1", "inspection_date": "2026-10-30",
        "sources": sources, "images": images, "algebra_name_scans": scans,
        "scope": {
            "ggufs": ["igbundle-llm/dist/base_model.gguf",
                      "igbundle_arc_deployment/qwen_arc_base_0.5B.gguf"],
            "inventory": "1577 directories visited in the workspace; nine selected research PNGs visually inspected.",
            "excluded": [
                "Git internals, environments, dependencies, caches, reparse-point directories",
                "Duplicate audit/falsification/publication worktrees",
                "Vendored llama.cpp trees (their model folders contain tokenizer fixtures)",
                "Multimodal dataset pictures and UI/vendor graphics from scientific-image review",
                "Model tensor payloads, pickle deserialization, full multi-GB checksums, inference/training",
                "Private cross-repository contents from public publication",
            ],
        },
        "limitations": [
            "Source and image hashes identify inspected files, not immutable source-to-generation bindings.",
            "Unseeded synthetic images were not reproduced pixel-for-pixel.",
            "Name searches are bounded evidence, not proof no algebra exists under other names or in other repos.",
            "No new model-performance, learned-hyperbolicity or hypercomplex-runtime result is asserted.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = export(args.archive)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
