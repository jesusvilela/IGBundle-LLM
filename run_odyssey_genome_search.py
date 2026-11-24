"""
Evolutionary search pipeline for Odyssey V3 recovery runs.

This script mutates a bounded "training genome", runs short pilot pushes from
the current promoted checkpoint, scores the resulting eval artifacts, and emits
the best genome plus a recommended command for the next 1000-step continuation.

The search is intentionally pragmatic:
  - It only mutates safe recovery knobs already supported by train_odyssey_v3.py.
  - It scores candidates with the same EOS / entropy / degeneration contract
    already used for checkpoint promotion.
  - It keeps every pilot run in an isolated output directory for auditability.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROMOTION_EOS_THRESHOLD = 0.64
PROMOTION_EVAL_S_THRESHOLD = 0.60
PROMOTION_LEAKAGE_THRESHOLD = 0.0
PROMOTION_DEGEN_THRESHOLD = 0.02
PROMOTION_BENCHMARK_FLOOR = 0.80


@dataclass(frozen=True)
class GenomeBounds:
    low: float
    high: float
    log_scale: bool = False


GENOME_BOUNDS: Dict[str, GenomeBounds] = {
    "base_lr": GenomeBounds(1e-5, 2e-4, log_scale=True),
    "fiber_lr": GenomeBounds(5e-4, 8e-3, log_scale=True),
    "geo_lambda_max": GenomeBounds(0.01, 0.15),
    "geo_ramp_steps": GenomeBounds(50.0, 400.0),
    "entropy_loss_scale": GenomeBounds(3.0, 12.0),
    "eos_ce_weight": GenomeBounds(0.5, 4.0),
    "eos_margin_weight": GenomeBounds(0.2, 2.5),
    "eos_stop_kl_weight": GenomeBounds(0.05, 1.0),
    "eos_norm_weight": GenomeBounds(0.01, 0.2),
}

INT_GENES = {"geo_ramp_steps"}


@dataclass
class OdysseyGenome:
    base_lr: float = 5e-5
    fiber_lr: float = 3e-3
    geo_lambda_max: float = 0.05
    geo_ramp_steps: int = 200
    entropy_loss_scale: float = 7.5
    eos_ce_weight: float = 2.0
    eos_margin_weight: float = 1.0
    eos_stop_kl_weight: float = 0.25
    eos_norm_weight: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)

    def command_args(self) -> List[str]:
        return [
            "--base-lr", f"{self.base_lr:.8g}",
            "--fiber-lr", f"{self.fiber_lr:.8g}",
            "--geo-lambda-max", f"{self.geo_lambda_max:.8g}",
            "--geo-ramp-steps", str(int(self.geo_ramp_steps)),
            "--entropy-loss-scale", f"{self.entropy_loss_scale:.8g}",
            "--eos-ce-weight", f"{self.eos_ce_weight:.8g}",
            "--eos-margin-weight", f"{self.eos_margin_weight:.8g}",
            "--eos-stop-kl-weight", f"{self.eos_stop_kl_weight:.8g}",
            "--eos-norm-weight", f"{self.eos_norm_weight:.8g}",
        ]

    def clamped(self) -> "OdysseyGenome":
        values: Dict[str, float] = {}
        for name, bounds in GENOME_BOUNDS.items():
            raw_value = getattr(self, name)
            value = min(bounds.high, max(bounds.low, float(raw_value)))
            if name in INT_GENES:
                value = int(round(value))
            values[name] = value
        return OdysseyGenome(**values)

    @classmethod
    def random(cls, rng: random.Random) -> "OdysseyGenome":
        values: Dict[str, float] = {}
        for name, bounds in GENOME_BOUNDS.items():
            if bounds.log_scale:
                lo = math.log10(bounds.low)
                hi = math.log10(bounds.high)
                sampled = 10 ** rng.uniform(lo, hi)
            else:
                sampled = rng.uniform(bounds.low, bounds.high)
            if name in INT_GENES:
                sampled = int(round(sampled))
            values[name] = sampled
        return cls(**values).clamped()

    def mutate(self, rng: random.Random, scale: float = 0.20) -> "OdysseyGenome":
        values = self.as_dict()
        for name, bounds in GENOME_BOUNDS.items():
            current = float(values[name])
            if bounds.log_scale:
                current_log = math.log10(current)
                span = math.log10(bounds.high) - math.log10(bounds.low)
                mutated = 10 ** (current_log + rng.gauss(0.0, max(1e-4, scale * span)))
            else:
                span = bounds.high - bounds.low
                mutated = current + rng.gauss(0.0, max(1e-4, scale * span))
            if name in INT_GENES:
                mutated = int(round(mutated))
            values[name] = mutated
        return OdysseyGenome(**values).clamped()

    @classmethod
    def crossover(
        cls,
        left: "OdysseyGenome",
        right: "OdysseyGenome",
        rng: random.Random,
    ) -> "OdysseyGenome":
        values: Dict[str, float] = {}
        for name in GENOME_BOUNDS:
            if rng.random() < 0.5:
                value = getattr(left, name)
            else:
                value = getattr(right, name)
            values[name] = value
        return cls(**values).clamped()


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def find_latest_eval_metrics(output_dir: Path) -> Optional[dict]:
    eval_files = sorted(output_dir.rglob("eval_metrics.json"), key=lambda p: p.stat().st_mtime)
    if not eval_files:
        return None
    return load_json(eval_files[-1])


def read_candidate_metrics(output_dir: Path) -> Optional[dict]:
    for name in ("promoted_checkpoint.json", "best_eos_checkpoint.json"):
        metrics = load_json(output_dir / name)
        if metrics is not None:
            return metrics
    return find_latest_eval_metrics(output_dir)


def score_candidate(metrics: dict) -> Tuple[float, Dict[str, float]]:
    eos_ratio = float(metrics.get("eos_ratio", 0.0))
    eval_s_mean = float(metrics.get("eval_S_mean", metrics.get("entropy_mean", 0.0)))
    leakage_rate = float(metrics.get("leakage_rate", 0.0))
    degeneration_rate = float(metrics.get("degeneration_rate", 0.0))
    benchmark_score = float(metrics.get("benchmark_score", 0.0))
    promotion_status = bool(metrics.get("promotion_status", False))

    score = 0.0
    score += 4.0 * eos_ratio
    score += 2.5 * min(1.0, eval_s_mean)
    score += 1.5 * benchmark_score
    score -= 8.0 * leakage_rate
    score -= 10.0 * degeneration_rate

    if eos_ratio >= PROMOTION_EOS_THRESHOLD:
        score += 1.0 + 2.0 * (eos_ratio - PROMOTION_EOS_THRESHOLD)
    else:
        score -= 4.0 * (PROMOTION_EOS_THRESHOLD - eos_ratio)

    if eval_s_mean >= PROMOTION_EVAL_S_THRESHOLD:
        score += 1.0 + 1.5 * min(0.5, eval_s_mean - PROMOTION_EVAL_S_THRESHOLD)
    else:
        score -= 6.0 * (PROMOTION_EVAL_S_THRESHOLD - eval_s_mean)

    if benchmark_score < PROMOTION_BENCHMARK_FLOOR:
        score -= 3.0 * (PROMOTION_BENCHMARK_FLOOR - benchmark_score)

    if leakage_rate > PROMOTION_LEAKAGE_THRESHOLD:
        score -= 2.0
    if degeneration_rate >= PROMOTION_DEGEN_THRESHOLD:
        score -= 2.0
    if promotion_status:
        score += 3.0

    breakdown = {
        "fitness": score,
        "eos_ratio": eos_ratio,
        "eval_S_mean": eval_s_mean,
        "benchmark_score": benchmark_score,
        "leakage_rate": leakage_rate,
        "degeneration_rate": degeneration_rate,
        "promotion_status": float(promotion_status),
    }
    return score, breakdown


def build_training_command(args: argparse.Namespace, genome: OdysseyGenome, output_dir: Path, max_steps: int) -> List[str]:
    cmd = [
        args.python_exe,
        args.train_script,
        "--stage", args.stage,
        "--model-id", args.model_id,
        "--checkpoint-dir", args.checkpoint_dir,
        "--resume-from", args.resume_from,
        "--output-dir", str(output_dir),
        "--max-steps", str(max_steps),
        "--checkpoint-every", str(args.checkpoint_every),
    ]
    if args.enable_vision:
        cmd.append("--enable-vision")
    if args.use_delta_fiber:
        cmd.append("--use-delta-fiber")
    if args.iacs_telemetry:
        cmd.append("--iacs-telemetry")
    cmd.extend(genome.command_args())
    return cmd


def run_candidate(
    args: argparse.Namespace,
    genome: OdysseyGenome,
    output_dir: Path,
    max_steps: int,
) -> Tuple[Optional[dict], List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_training_command(args, genome, output_dir, max_steps)
    if args.dry_run:
        return None, cmd

    if not args.force and read_candidate_metrics(output_dir) is not None:
        return read_candidate_metrics(output_dir), cmd

    start = time.time()
    proc = subprocess.run(cmd, cwd=args.workdir, check=False)
    duration = time.time() - start

    metrics = read_candidate_metrics(output_dir)
    if metrics is not None:
        metrics["wall_clock_seconds"] = round(duration, 2)
        metrics["return_code"] = proc.returncode
    elif proc.returncode == 0:
        metrics = {"return_code": 0, "wall_clock_seconds": round(duration, 2)}
    else:
        metrics = {"return_code": proc.returncode, "wall_clock_seconds": round(duration, 2)}
    return metrics, cmd


def initial_population(args: argparse.Namespace, rng: random.Random) -> List[OdysseyGenome]:
    base = OdysseyGenome().clamped()
    genomes = [base]
    while len(genomes) < args.population:
        genomes.append(base.mutate(rng, scale=args.mutation_scale))
    return genomes


def evolve_population(
    ranked: List[dict],
    args: argparse.Namespace,
    rng: random.Random,
) -> List[OdysseyGenome]:
    elites = [OdysseyGenome(**entry["genome"]) for entry in ranked[: args.elites]]
    next_gen = list(elites)
    while len(next_gen) < args.population:
        if len(elites) == 1:
            child = elites[0].mutate(rng, scale=args.mutation_scale)
        else:
            left, right = rng.sample(elites, 2)
            child = OdysseyGenome.crossover(left, right, rng).mutate(rng, scale=args.mutation_scale)
        next_gen.append(child)
    return next_gen


def summarize_best_command(args: argparse.Namespace, genome: OdysseyGenome) -> List[str]:
    output_dir = Path(args.final_output_dir)
    return build_training_command(args, genome, output_dir, args.full_push_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Odyssey V3 genome search runner")
    parser.add_argument("--stage", default="alignment-eos")
    parser.add_argument("--model-id", default="h:/LLM-MANIFOLD/igbundle_qwen7b_cp600")
    parser.add_argument("--checkpoint-dir", default="igbundle_odyssey_v3")
    parser.add_argument("--resume-from", default="checkpoint-alignment-eos-1400")
    parser.add_argument("--train-script", default="train_odyssey_v3.py")
    parser.add_argument("--workdir", default=os.getcwd())
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--pilot-steps", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--full-push-steps", type=int, default=1000)
    parser.add_argument("--output-root", default="odyssey_genome_search")
    parser.add_argument("--final-output-dir", default="igbundle_odyssey_v3_iter_genome")
    parser.add_argument("--mutation-scale", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-vision", action="store_true")
    parser.add_argument("--use-delta-fiber", action="store_true")
    parser.add_argument("--iacs-telemetry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.population < 1:
        raise SystemExit("--population must be >= 1")
    if args.elites < 1:
        raise SystemExit("--elites must be >= 1")
    if args.elites > args.population:
        raise SystemExit("--elites cannot exceed --population")

    rng = random.Random(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    history: List[dict] = []
    population = initial_population(args, rng)
    best_entry: Optional[dict] = None

    for generation in range(args.generations):
        results: List[dict] = []
        for candidate_idx, genome in enumerate(population):
            run_dir = output_root / f"gen_{generation:02d}" / f"cand_{candidate_idx:02d}"
            metrics, cmd = run_candidate(args, genome, run_dir, args.pilot_steps)

            if metrics is None:
                fitness = float("nan")
                breakdown = {"fitness": float("nan")}
            else:
                fitness, breakdown = score_candidate(metrics)

            record = {
                "generation": generation,
                "candidate": candidate_idx,
                "genome": genome.as_dict(),
                "output_dir": str(run_dir),
                "command": cmd,
                "metrics": metrics,
                "score": breakdown,
            }
            results.append(record)
            history.append(record)

        ranked = sorted(
            results,
            key=lambda entry: entry["score"].get("fitness", float("-inf")),
            reverse=True,
        )
        if ranked and (best_entry is None or ranked[0]["score"]["fitness"] > best_entry["score"]["fitness"]):
            best_entry = ranked[0]

        if generation + 1 < args.generations:
            population = evolve_population(ranked, args, rng)

    summary = {
        "search_config": {
            "population": args.population,
            "generations": args.generations,
            "pilot_steps": args.pilot_steps,
            "full_push_steps": args.full_push_steps,
            "resume_from": args.resume_from,
            "checkpoint_dir": args.checkpoint_dir,
            "seed": args.seed,
        },
        "best": best_entry,
        "history": history,
    }

    if best_entry is not None:
        best_genome = OdysseyGenome(**best_entry["genome"]).clamped()
        summary["recommended_full_push_command"] = summarize_best_command(args, best_genome)
        with (output_root / "best_genome.json").open("w", encoding="utf-8") as handle:
            json.dump(best_genome.as_dict(), handle, indent=2)

    with (output_root / "search_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if best_entry is not None:
        print("Best genome:")
        print(json.dumps(best_entry["genome"], indent=2))
        print("\nRecommended 1000-step push:")
        print(" ".join(summary["recommended_full_push_command"]))
    else:
        print("No completed pilot runs were available.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
