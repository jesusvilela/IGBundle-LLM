# IGBundle / ManifoldGL research status

IGBundle explores information-geometric operations and fiber-structured adapters as a possible inductive bias for language models. This is a research hypothesis, not a validated improvement. Bundle terminology describes the intended parameterization; it does not demonstrate that a trained model has learned semantic hierarchy or a fiber bundle.

## What has been measured

- **October 27, 2026 local subset evaluation:** custom weighted index 61.1 for the base, 60.0 for the merged cp300 checkpoint, and 64.7 for a reconstructed adapter composite. No individual paired benchmark comparison was statistically significant. Subsets ranged from 8 to 98 examples; the composite was not the original trained checkpoint. [Per-benchmark summary](docs/benchmark-results/aa-index-2026-10-27.csv).
- **Artifact integrity:** the exported cp300 weights in that run did not contain the runtime geometric adapter. The reconstructed arm grafted a surviving base-trained adapter onto cp300 and used a different loading/quantization path; it does not isolate a geometric effect.
- **Fixed-operator audit:** 6/6 checks passed for the configuration in [`A_faithfulness_audit.json`](experiments/falsification/results/A_faithfulness_audit.json). This verifies only the tested operators.
- **Curvature:** the default conformal factor is fixed, making the historical readout weight-invariant. Do not describe it as learned curvature; see [`draft_paper_falsification.md`](draft_paper_falsification.md).
- **Paired token-NLL probe (2026-09-20):** on 24 held-out examples × 64 tokens, mean NLL was 2.3188 for the frozen base, 1.9352 for geometric adapter A, and 0.8187 for vanilla-LoRA control B. B had lower NLL on all 24 examples. [Per-example scores](docs/benchmark-results/matched-token-nll-2026-09-20.csv).
- **Comparison limits:** A used 4,160,713 trainable parameters and one geometric layer; B used 585,728 parameters and 13 LoRA layers. This probe does not isolate a geometric effect and is not task accuracy.
- **Historical ARC-Challenge:** zero-shot `acc_norm` was 54.86% for both the adapter and base on 1,172 items each.
- **Historical ARC-AGI:** 1/100 exact solves; Model-First Reasoning format compliance was 0/100. The latter is a prompt-format metric, not geometric faithfulness. These results have no matched base comparison.

The previously advertised +131.5% ARC-AGI gain and 94.2% “MFR” figure have no support in the measured artifacts. Older single-model benchmark scores and efficiency claims are not evidence of improvement without suitable matched controls. No general downstream performance advantage is currently established.

## Research framing

The approach remains open to falsification. A stronger comparison should match parameter budget and layer placement, hold data and training conditions constant, use multiple seeds, and report task-level results and raw outputs. Fixed-kernel tests, geometry diagnostics, and language-task evaluations answer different questions and must not be conflated.

See the [README](README.md), [site evidence page](docs/wiki.html), [October per-benchmark summary](docs/benchmark-results/aa-index-2026-10-27.csv), [research-to-code ledger](docs/research_to_code_evidence_ledger.md), and [source repository](https://github.com/jesusvilela/IGBundle-LLM).
