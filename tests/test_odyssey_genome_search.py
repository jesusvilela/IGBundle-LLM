import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_odyssey_genome_search import (
    GENOME_BOUNDS,
    OdysseyGenome,
    build_training_command,
    score_candidate,
)


class DummyArgs:
    python_exe = "python"
    train_script = "train_odyssey_v3.py"
    stage = "alignment-eos"
    model_id = "model"
    checkpoint_dir = "igbundle_odyssey_v3"
    resume_from = "checkpoint-alignment-eos-1400"
    checkpoint_every = 50
    enable_vision = False
    use_delta_fiber = False
    iacs_telemetry = False


def test_genome_clamp_respects_bounds():
    genome = OdysseyGenome(
        base_lr=1.0,
        fiber_lr=0.0,
        geo_lambda_max=-1.0,
        geo_ramp_steps=9999,
        entropy_loss_scale=99.0,
        eos_ce_weight=-3.0,
        eos_margin_weight=99.0,
        eos_stop_kl_weight=-2.0,
        eos_norm_weight=9.0,
    ).clamped()

    values = genome.as_dict()
    for name, bounds in GENOME_BOUNDS.items():
        assert bounds.low <= float(values[name]) <= bounds.high
    assert isinstance(genome.geo_ramp_steps, int)


def test_mutation_stays_bounded_and_changes_values():
    rng = random.Random(7)
    genome = OdysseyGenome()
    mutated = genome.mutate(rng, scale=0.35)

    assert mutated != genome
    for name, bounds in GENOME_BOUNDS.items():
        value = getattr(mutated, name)
        assert bounds.low <= float(value) <= bounds.high


def test_crossover_uses_parent_genes():
    rng = random.Random(11)
    left = OdysseyGenome(base_lr=1e-5, fiber_lr=1e-3, geo_ramp_steps=100)
    right = OdysseyGenome(base_lr=2e-4, fiber_lr=8e-3, geo_ramp_steps=300)

    child = OdysseyGenome.crossover(left, right, rng)

    assert child.base_lr in {left.base_lr, right.base_lr}
    assert child.fiber_lr in {left.fiber_lr, right.fiber_lr}
    assert child.geo_ramp_steps in {left.geo_ramp_steps, right.geo_ramp_steps}


def test_score_candidate_rewards_promotion_ready_metrics():
    strong_score, strong = score_candidate(
        {
            "eos_ratio": 0.70,
            "eval_S_mean": 0.85,
            "benchmark_score": 0.9,
            "leakage_rate": 0.0,
            "degeneration_rate": 0.0,
            "promotion_status": True,
        }
    )
    weak_score, weak = score_candidate(
        {
            "eos_ratio": 0.40,
            "eval_S_mean": 0.10,
            "benchmark_score": 0.3,
            "leakage_rate": 0.1,
            "degeneration_rate": 0.2,
            "promotion_status": False,
        }
    )

    assert strong["promotion_status"] == 1.0
    assert weak["promotion_status"] == 0.0
    assert strong_score > weak_score


def test_build_training_command_includes_genome_overrides():
    cmd = build_training_command(
        DummyArgs(),
        OdysseyGenome(base_lr=2.5e-5, geo_ramp_steps=123),
        Path("out"),
        max_steps=250,
    )

    joined = " ".join(cmd)
    assert "--base-lr 2.5e-05" in joined
    assert "--geo-ramp-steps 123" in joined
    assert "--max-steps 250" in joined
    assert "--output-dir out" in joined
