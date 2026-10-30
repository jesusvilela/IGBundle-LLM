# IGBundle: Fiber-Structured Geometric Adapters for Language Models

**A research prototype exploring whether information-geometric operations and fiber-structured adapters can provide a useful inductive bias for language models.**

[![License: All Rights Reserved](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)
![Status: Experimental Research](https://img.shields.io/badge/Status-Experimental%20Research-purple.svg)

> **Evidence status:** The geometric inductive-bias idea remains a valid research hypothesis, but the available experiments do **not** establish a general task-performance advantage, learned hyperbolicity, or training/inference efficiency gains. Previously advertised accuracy, “MFR,” ablation, and efficiency claims are retracted. See the [correction notice](docs/correction-notice.md), [evidence ledger](docs/research_to_code_evidence_ledger.md), and [published benchmark summaries](docs/benchmark-results/).

## Research design

IGBundle explores Poincare-ball coordinates, categorical fiber components, and geometric or dynamical operators as a possible structure for adapting language models. These describe the intended parameterization; they are not evidence that a trained model has learned semantic hierarchy, a fiber bundle, sheaf structure, or hyperbolic geometry.

The default curvature readout is weight-invariant because its conformal factor is fixed. It must not be presented as learned curvature. Fixed-operator checks, geometry diagnostics, and downstream task evaluation answer different questions and are reported separately.

## What the evidence currently shows

### October 2026 local multi-benchmark evaluation

A 27 October local run compared an exported Qwen2.5-7B checkpoint (cp300), the base model, and a reconstructed composite with runtime IGBundle adapters. It used greedy decoding on fixed, small subsets (8–98 examples per task). Its “intelligence index” is a custom weighted average, not a standardized benchmark:

| Variant | Custom weighted index | Scope |
|:---|---:|:---|
| Base Qwen2.5-7B | 61.1 | Q4_K_M GGUF |
| Merged cp300 checkpoint | 60.0 | Q4_K_M GGUF |
| Reconstructed adapter composite | 64.7 | Separate NF4/bf16 loading path |

No individual paired task comparison was statistically significant. The reconstructed composite had higher point estimates than cp300 on 5/6 tasks, but every reported McNemar p-value was at least 0.63. These small subsets cannot establish equivalence or a broad benefit.

**Checkpoint caveat:** the exported cp300 weights did not contain the runtime geometric adapter. The reconstructed arm grafted a surviving adapter trained on the base model onto cp300; it was not the original trained checkpoint. Model assembly and quantization also differed. This exploratory, post-hoc reconstruction does not isolate a geometric effect. The [per-task summary](docs/benchmark-results/aa-index-2026-10-27.csv) includes sample counts, scores, weights, parse failures, and paired p-values.

| Task | n | Base | Merged cp300 | Reconstructed composite | McNemar p, composite vs cp300 |
|:---|---:|---:|---:|---:|---:|
| MMLU-Pro | 98 | 53.1% | 48.0% | 51.0% | 0.68 |
| ARC-Challenge | 60 | 90.0% | 88.3% | 91.7% | 0.69 |
| MATH-500 | 40 | 40.0% | 45.0% | 50.0% | 0.73 |
| GSM8K | 30 | 83.3% | 76.7% | 80.0% | 1.00 |
| IFBench-lite | 16 | 68.8% | 68.8% | 81.2% | 0.63 |
| Degeneration probes | 8 | 75.0% | 75.0% | 75.0% | 1.00 |

### September 2026 paired token-NLL probe

On 24 held-out examples with 64 tokens scored per example, mean token negative log-likelihood was 2.3188 for the frozen base, 1.9352 for geometric adapter A, and 0.8187 for vanilla-LoRA control B. B had lower NLL on all 24 examples. The mean paired difference `A − B` was +1.1165 (95% paired t interval `[0.9144, 1.3186]`; `p = 5.83 × 10⁻¹¹`).

This is a small language-modeling probe, not task accuracy. A had 4,160,713 trainable parameters and one geometric layer; B had 585,728 parameters and 13 LoRA layers. The comparison does not isolate geometry, layer coverage, or objective. [Per-example scores](docs/benchmark-results/matched-token-nll-2026-09-20.csv) are published without prompt text.

### Fixed-operator audit and historical task evaluations

- The recorded local operator audit reports **6/6 checks passed** for its tested configuration. It does not measure learned geometry or task quality. [Audit data](docs/benchmark-results/faithfulness-audit-A.json).
- Historical zero-shot ARC-Challenge `acc_norm` was **54.86% for both** an adapter checkpoint and the base (1,172 test items per run). This is a separate protocol and dataset size from the October subset run. [Adapter results](docs/benchmark-results/arc-challenge-adapter-2026-01.json) · [Base results](docs/benchmark-results/arc-challenge-base-2026-01.json).
- A separate historical ARC-AGI evaluation solved **1/100 tasks (1%)**; Model-First Reasoning format compliance was **0/100**. The latter is a prompt-format metric, not geometric faithfulness. [Aggregate data](docs/benchmark-results/arc-agi-summary.json).

The historical curvature value near -5.63 comes from the fixed conformal component, not learned hyperbolicity. See the [falsification analysis](docs/correction-notice.md) and its source derivation in the repository's research record.

## Project structure

```
src/igbundle/
├── geometry/          # Riemannian, hyperbolic, Poincare, KAN manifold
├── modules/           # Geometric adapter, losses, attention, vision
├── dynamics/          # Hamiltonian and related dynamics
├── fibers/            # Fiber state, constraints, and executors
├── steering/          # Geometric Steering Probe prototypes
├── optimization/      # Optimizer prototypes
└── training/          # Training utilities and losses

tests/                 # Unit and integration tests
configs/               # Research configurations
docs/                  # GitHub Pages and evidence summaries
```

## Getting the code

```bash
git clone https://github.com/jesusvilela/IGBundle-LLM.git
cd IGBundle-LLM
```

This repository is experimental code, not a supported inference product. Existing hosted checkpoints are historical artifacts; verify their source/configuration compatibility and model-license terms before use. No model download, training run, or hardware requirement is implied by cloning this repository.

## Research standard

The hypothesis remains open. A stronger causal test should match parameter budget and layer placement, isolate the geometric operator/objective, use a shared split and training budget with multiple seeds, and publish exact checkpoint provenance and task-level outputs. Geometry diagnostics alone cannot establish language-model benefit.

## Citation

```bibtex
@misc{vilela2025igbundle,
    title   = {IGBundle: Fiber-Structured Geometric Adapters for Language Models},
    author  = {Vilela Jato, Jes{\'u}s},
    year    = {2025},
    url     = {https://github.com/jesusvilela/IGBundle-LLM}
}
```

## License

All rights reserved. See [LICENSE](LICENSE).

*© 2025–2026 Jesús Vilela Jato*
