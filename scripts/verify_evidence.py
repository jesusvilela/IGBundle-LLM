"""Recompute published evidence arithmetic and check static-site links, offline.

Uses only the Python standard library. Success verifies consistency of recorded
measurements, not the historical graders, model provenance, or scientific efficacy.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"mmlupro": 0.30, "arc": 0.10, "math": 0.25, "gsm8k": 0.10, "ifbench": 0.15, "probes": 0.10}
VARIANTS = ("merged_cp300", "base_qwen", "reconstructed_adapter_composite")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def exact_mcnemar(first_only, second_only):
    n = first_only + second_only
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(first_only, second_only) + 1)) / 2**n)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for key, value in attrs if key in ("href", "src") and value)


def check_site_links(root):
    checked = 0
    for page in (root / "docs").glob("*.html"):
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for url in parser.urls:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = (page.parent / unquote(parts.path)).resolve()
            require(target.is_relative_to((root / "docs").resolve()), f"Link escapes site root: {page.name}: {url}")
            require(target.is_file(), f"Missing static-site target: {page.name}: {url}")
            checked += 1
    return checked


def verify(root=ROOT):
    data = root / "docs" / "benchmark-results"
    nll = read_csv(data / "matched-token-nll-2026-09-20.csv")
    require(len(nll) == 24, "September probe must contain 24 pairs")
    for field in ("sample", "dataset_index", "text_sha256"):
        require(len({row[field] for row in nll}) == 24, f"Duplicate {field} in NLL data")
    differences = []
    for row in nll:
        values = [float(row[key]) for key in ("base_nll", "geometric_A_nll", "vanilla_LoRA_B_nll")]
        require(all(math.isfinite(value) and value >= 0 for value in values), "Invalid NLL")
        require(int(row["token_count"]) == 64, "Unexpected token count")
        delta = values[1] - values[2]
        require(math.isclose(delta, float(row["geometric_minus_control_nll"]), abs_tol=1e-12), "NLL delta mismatch")
        differences.append(delta)
    means = {key: statistics.mean(float(row[key]) for row in nll)
             for key in ("base_nll", "geometric_A_nll", "vanilla_LoRA_B_nll")}
    delta_mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(24)
    # Fixed published probe: two-sided t_0.975 with 23 degrees of freedom.
    half_width = 2.0686576104190406 * se
    wins = sum(delta > 0 for delta in differences)
    nonzero = sum(delta != 0 for delta in differences)
    sign_p = min(1.0, 2 * sum(math.comb(nonzero, k) for k in range(min(wins, nonzero - wins) + 1)) / 2**nonzero)
    for key, expected in zip(means, (2.3188, 1.9352, 0.8187)):
        require(abs(means[key] - expected) < 0.00005, f"Published {key} mismatch")
    require(wins == 24, "Published control-win count mismatch")
    require(abs(delta_mean - 1.1165) < 0.00005, "Published paired delta mismatch")

    paired = read_csv(data / "aa-paired-outcomes-2026-10-27.csv")
    aggregates = read_csv(data / "aa-index-2026-10-27.csv")
    require(len(paired) == 252, "October paired rows must cover 252 items")
    require(len({(row["benchmark"], row["item_id"]) for row in paired}) == len(paired), "Duplicate October IDs")
    require({row["benchmark"] for row in paired} == set(WEIGHTS), "Unexpected benchmark set")
    require(all(row[variant] in ("0", "1") for row in paired for variant in VARIANTS), "Non-binary grade")
    require(len(aggregates) == 21, "Unexpected aggregate row count")
    require(len({(row["variant"], row["benchmark"]) for row in aggregates}) == 21, "Duplicate aggregates")
    summary_lookup = {(row["variant"], row["benchmark"]): row for row in aggregates}
    october = {}
    for variant in VARIANTS:
        index = 0.0
        scores = {}
        for benchmark, weight in WEIGHTS.items():
            rows = [row for row in paired if row["benchmark"] == benchmark]
            recorded = summary_lookup[(variant, benchmark)]
            accuracy = 100 * sum(int(row[variant]) for row in rows) / len(rows)
            require(len(rows) == int(recorded["sample_size"]), "October sample-size mismatch")
            require(abs(accuracy - float(recorded["accuracy_percent"])) <= 0.0051, "October accuracy mismatch")
            require(float(recorded["weighted_index_weight"]) == weight, "October weight mismatch")
            score = {"n": len(rows), "accuracy_percent": accuracy}
            if variant != "merged_cp300":
                first_only = sum(row[variant] == "1" and row["merged_cp300"] == "0" for row in rows)
                second_only = sum(row[variant] == "0" and row["merged_cp300"] == "1" for row in rows)
                p = exact_mcnemar(first_only, second_only)
                require(abs(p - float(recorded["mcnemar_p_vs_merged_cp300"])) <= 0.0051, "McNemar p mismatch")
                score.update(first_only=first_only, merged_only=second_only, exact_mcnemar_p=p)
            scores[benchmark] = score
            index += weight * accuracy
        require(abs(index - float(summary_lookup[(variant, "weighted_index")]["accuracy_percent"])) < 0.05,
                "Weighted index mismatch")
        october[variant] = {"weighted_index": index, "benchmarks": scores}

    rescored = json.loads((data / "arc-agi-rescore.json").read_text(encoding="utf-8"))
    require(len(rescored) == 100 and len({row["task_id"] for row in rescored}) == 100, "ARC historical row count")
    historical_arc = json.loads((data / "arc-agi-summary.json").read_text(encoding="utf-8"))
    require(sum(row["recorded_correct"] for row in rescored) == historical_arc["exact_solves"], "Historical ARC total mismatch")
    arc_challenge = {}
    for arm in ("adapter", "base"):
        record = json.loads((data / f"arc-challenge-{arm}-2026-01.json").read_text(encoding="utf-8"))
        score = record["results"]["arc_challenge"]["acc_norm,none"]
        require(abs(score * 100 - 54.86) < 0.005, "Historical ARC-Challenge score mismatch")
        require(record["n-samples"]["arc_challenge"]["effective"] == 1172, "Historical ARC-Challenge sample count")
        arc_challenge[arm] = score
    return {
        "scope": "Offline arithmetic and link checks only; not a new model benchmark.",
        "september": {"n": 24, "means": means, "control_wins": wins, "mean_A_minus_B": delta_mean,
                      "paired_t_95_interval": [delta_mean - half_width, delta_mean + half_width],
                      "paired_t_statistic": delta_mean / se, "two_sided_sign_p": sign_p},
        "october": october,
        "historical_arc_first_pair_rescore": {
            "n": len(rescored), "recorded_correct": sum(row["recorded_correct"] for row in rescored),
            "strict_first_pair_correct": sum(row["strict_first_pair_correct"] for row in rescored),
            "errors": sum(row["recorded_error"] for row in rescored),
            "scope": "First test pair in saved generations, not all-test-pair ARC task solving.",
        },
        "historical_arc_challenge": arc_challenge,
        "local_site_links_checked": check_site_links(root),
        "artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(data.iterdir()) if path.suffix in (".csv", ".json", ".txt")
                            and path.name != "verification.json"},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered)
