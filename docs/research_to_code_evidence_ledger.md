# Research-to-code evidence ledger

Research documents generate hypotheses; implementation and operator tests establish local behavior; matched task experiments are required to establish empirical effects. None substitutes for another.

| Claim | Evidence | Conclusion |
|---|---|---|
| Fiber-structured geometric adapters improve reasoning | The October 2026 local multi-benchmark comparison found no significant individual paired task differences; its custom index is exploratory. The September 2026 token-NLL probe favored its vanilla-LoRA control on all 24 paired examples. | Open hypothesis; no general advantage established. |
| The October reconstructed composite improved the model | Its custom index was 64.7 vs 60.0 for cp300 and 61.1 for the base; per-task paired McNemar p-values vs cp300 were 0.63–1.00. The composite was assembled post hoc from an adapter trained on the base, not the original cp300 companion; model loading/quantization also differed. | Descriptive point estimates only; not causal or independent confirmation. |
| The exported cp300 model executes the geometric adapter | The October audit reports the merged checkpoint's served tensor index contains Qwen tensors only; the geometric adapter was a separate runtime module and was absent from the exported model. | Do not describe the merged cp300 evaluation as an inference-time geometric-adapter evaluation. |
| Training learned curvature near -5.63 | The default conformal factor is fixed and its curvature readout is weight-invariant. | Retracted as learned-curvature evidence; see `correction-notice.md`. |
| Fixed geometric operations have local structure | The recorded audit reports 6/6 checks passed for the tested configuration. | Local operator diagnostics only; not task performance or learned geometry. |
| Historical ARC results show a gain | ARC-Challenge zero-shot `acc_norm` was 54.86% for both adapter and base on 1,172 items each; ARC-AGI exact solve was 1/100 with no matched base comparison. | No gain demonstrated. |
| Historical natural-gradient/geometry terms reduce training or inference cost | No suitable matched, reproducible comparison is available in the public evidence artifacts. | Withdrawn; do not report efficiency gains or overheads as established results. |

## Published data

- [October 27, 2026 per-benchmark summary](benchmark-results/aa-index-2026-10-27.csv)
- [September 20, 2026 paired token-NLL rows](benchmark-results/matched-token-nll-2026-09-20.csv)
- [Faithfulness audit summary](benchmark-results/faithfulness-audit-A.json)
- [Historical ARC-Challenge adapter](benchmark-results/arc-challenge-adapter-2026-01.json) and [base](benchmark-results/arc-challenge-base-2026-01.json)
- [Historical ARC-AGI summary](benchmark-results/arc-agi-summary.json)

The October and September raw prompt/generation logs are not included here. The published CSVs contain aggregate scores or prompt-free paired measurements; reproduce claims only within the protocols and limitations stated in the correction notice and README.
