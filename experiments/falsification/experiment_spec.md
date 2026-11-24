# Falsification Experiment: Geometry vs. Layer Count

> **Execution correction (2026-09-20):** The saved arms are not parameter
> matched. Recorded trainable counts are A=4,160,713, B=585,728, C=2,342,912.
> A is a geometric adapter without the stated rank-8 LoRA arm; B/C are LoRA.
> Treat this as a matched data/seed/steps comparison with architecture and layer
> coverage confounded, not as a clean geometry-only ablation.

## Null Hypothesis (stated before results)

> Multi-layer vanilla LoRA (layers 8–20, no geometric losses) matches or exceeds
> single-layer geometric LoRA (layer 12, full IGBundle losses) on task benchmarks.
> If we cannot reject H₀ at p < 0.05, geometry does not have detectable signal at this scale.

## Configurations

| Config | Layers | Rank | Trainable Params | Geometric Losses |
|--------|--------|------|-----------------|-----------------|
| A: geo-single | 12 only | N/A | 4,160,713 | Full (curvature, entropy, fiber, consistency) |
| B: vanilla-multi-small | 8–20 (13 layers) | 2 | 585,728 | None (LLM loss only) |
| C: vanilla-multi-larger | 8–20 (13 layers) | 8 | 2,342,912 | None (LLM loss only) |

- Config B is smaller than A by about 7.1x, not parameter-matched.
- Config C is smaller than A by about 1.8x and shares B's layer coverage.
- The executed comparison answers whether these complete training recipes differ;
  it cannot attribute the difference to geometry alone.

## Base Model

- Qwen2.5-7B-Instruct, NF4 quantization via bitsandbytes
- PEFT LoRA on Q, K, V, O projections
- Same tokenizer, same ChatML template

## Training

- Dataset: same as checkpoint-3000 training data
- Steps: 1000 (all configs)
- Learning rate: 2e-4 (cosine, 100-step warmup)
- Batch size: 4 (gradient accumulation 4 = effective 16)
- Optimizer: AdamW, weight_decay=0.01
- Seed: 42 (fixed for reproducibility)

## Eval Battery

### Tier 1: Standard Benchmarks (lm-evaluation-harness)

| Benchmark | Subset Size | Metric | Source |
|-----------|------------|--------|--------|
| GSM8K | 200 problems | Exact match accuracy | lm-evaluation-harness |
| MMLU | 200 questions (balanced across subjects) | Accuracy | lm-evaluation-harness |
| ARC-Easy | 200 questions | Accuracy | lm-evaluation-harness |

- Each prompt: **10 runs at temperature 0.6**
- Report: **mean ± std** per benchmark per config
- Statistical test: paired t-test (A vs B, A vs C) per benchmark
- Significance threshold: p < 0.05

### Tier 2: ARC-AGI Reasoning (existing eval_arc.py infrastructure)

| Benchmark | Tasks | Metric | Source |
|-----------|-------|--------|--------|
| ARC-AGI-2 eval | All 120 eval tasks | Exact grid match accuracy + Wilson 95% CI | data/ARC-AGI-2/data/evaluation/ |

Note: `data/arc_agi/` and `data/ARC-AGI-2/data/evaluation/` are identical (same 120 files).
No separate v1 eval set exists locally. Use the full 120-task eval set for all configs.

- 1-shot prompting (task's own train pairs as examples, per existing eval_arc.py)
- Temperature 0.0 (greedy, deterministic — grid tasks need exact output)
- Run once per config (deterministic at temp 0)
- **Existing baseline**: 1% accuracy on 100-task subset (arc_evaluation_results.json) — must reproduce on full 120
- Also run against **bare Qwen2.5-7B** (no adapter) as Config D (zero-cost baseline)
- If bare base ≈ geometric adapter on ARC-AGI-2, the adapter adds no reasoning signal

### Tier 3: Manifold Faithfulness Audit

**CRITICAL**: The "100% manifold faithfulness" claim was never computed at eval time.
`train_odyssey_v3.py:1168` hardcodes `constraint_score: 1.0` in the GSP telemetry dict.

Actual faithfulness must be **measured**, not asserted. For Config A only:

| Metric | Computation | Pass Criterion |
|--------|-------------|----------------|
| Curvature stability | K variance across 100 eval batches | σ(K) < 1.0 |
| Fiber section entropy | S computed from actual fiber_sections at eval | S ≠ ln(K) ± 0.01 (not frozen at uniform) |
| Fiber diversity | JSD(sections, uniform) across eval batches | JSD > 0.01 (not degenerate) |
| Parallel transport fidelity | ||PT(v, x→y) - v||/||v|| for 100 random pairs | Ratio < 0.5 (not identity) |
| Geodesic triangle inequality | d(x,z) ≤ d(x,y) + d(y,z) for 100 random triples | 100% satisfaction |
| GENERIC energy conservation | |E(t+dt) - E(t)|/|E(t)| across integration steps | Drift < 1% per step |

If any metric fails, "100% manifold faithfulness" is falsified regardless of task performance.

## Configurations Added

| Config | Layers | Rank | Trainable Params | Geometric Losses | Purpose |
|--------|--------|------|-----------------|-----------------|---------|
| D: bare-base | None | 0 | 0 | None | Raw Qwen2.5-7B baseline (no adapter at all) |

Config D is zero-cost — just load the base model and run eval. Establishes whether ANY adapter helps on ARC-AGI, or if 1% was already the base model's ceiling.

## File Ownership

| File | Owner |
|------|-------|
| `train_baseline_lora.py` | Claude Code |
| `train_geo_lora.py` (stripped-down rerun of config A) | Claude Code |
| `eval_runner.py` (Tier 1 benchmarks) | Gemini |
| `eval_arc_falsification.py` (Tier 2 ARC-AGI) | Gemini |
| `audit_faithfulness.py` (Tier 3 manifold audit) | Claude Code |
| `experiments/falsification/results/` | Shared (write-once, read-both) |

## Output Schemas

### Tier 1: `results/{config}_benchmarks.json`

```json
{
  "config": "A|B|C|D",
  "checkpoint_path": "experiments/falsification/checkpoints/{config}/",
  "eval_timestamp": "ISO-8601",
  "benchmarks": {
    "gsm8k":    {"mean": 0.0, "std": 0.0, "n_runs": 10, "n_problems": 200},
    "mmlu":     {"mean": 0.0, "std": 0.0, "n_runs": 10, "n_problems": 200},
    "arc_easy": {"mean": 0.0, "std": 0.0, "n_runs": 10, "n_problems": 200}
  },
  "trainable_params": 0,
  "training_steps": 1000,
  "wall_time_seconds": 0
}
```

### Tier 2: `results/{config}_arc_agi.json`

```json
{
  "config": "A|B|C|D",
  "checkpoint_path": "...",
  "eval_timestamp": "ISO-8601",
  "arc_agi_2": {
    "accuracy": 0.0,
    "correct": 0,
    "total": 120,
    "wilson_ci_95": [0.0, 0.0]
  },
  "temperature": 0.0,
  "n_shots": 1,
  "details": []
}
```

### Tier 3: `results/A_faithfulness_audit.json` (Config A only)

```json
{
  "config": "A",
  "eval_timestamp": "ISO-8601",
  "curvature_stability": {"K_mean": 0.0, "K_std": 0.0, "pass": false},
  "fiber_entropy":       {"S_mean": 0.0, "S_at_uniform": 2.77, "frozen": false, "pass": false},
  "fiber_diversity":     {"jsd_mean": 0.0, "pass": false},
  "parallel_transport":  {"mean_displacement_ratio": 0.0, "pass": false},
  "geodesic_triangle":   {"violations": 0, "total_triples": 100, "pass": false},
  "generic_energy":      {"max_drift_pct": 0.0, "pass": false},
  "overall_faithfulness": false
}
```

## Timeline

- **2026-03-25**: Spec agreed, training scripts written, eval runner written
- **2026-03-25 evening**: Launch configs A, B, C (overnight training)
- **2026-03-26 morning**: Eval runs, results collected
- **2026-03-26 afternoon**: Statistical analysis, decision
- **Hard stop**: If not launched by EOD 2026-03-26, defer until after SMAC (2026-03-30)
