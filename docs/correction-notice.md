# IGBundle empirical-claims correction

**Status: 30 October 2026.** This notice supersedes unsupported empirical claims in older README, website, thesis, and model-card text. The geometric inductive-bias idea remains an open research hypothesis; the measurements below do not establish that it is ineffective in general.

## Withdrawn claims

Do not cite the following older figures as validated results:

- ARC-AGI accuracy of 28.7% versus a 12.4% baseline, or a +131.5% relative improvement.
- 94.2% or 100% “MFR” or any interpretation of such values as learned geometric faithfulness.
- A “learned curvature” of σ ≈ 2.2. In the adapter, σ is a per-component Gaussian scale (`log_sigma`), not a curvature; a nonzero value does not show that the data prefers a non-Euclidean representation.
- Ablation effects, statistical significance, curvature convergence to -1, or claims that the model learned semantic hyperbolicity.
- Claims of 30% faster convergence, 15% training overhead, 4% inference latency overhead, or specified VRAM savings.
- Unmatched single-model benchmark scores presented as evidence that an adapter preserved or improved base-model capabilities.

The historical ARC-AGI artifact records 1 success in 100 tasks and 0/100 Model-First Reasoning format-compliance cases. Inspection of the evaluator shows it scored only the first test pair and allowed substring matching. It is not an all-test-pair exact-task solve rate. Prompt-free strict rescoring of the saved first-pair generations is now published separately. The ARC-Challenge zero-shot artifacts record `acc_norm = 54.86%` for both an adapter checkpoint and its base on 1,172 test items each. The protocols differ; neither supports the older ARC-AGI gain claim.

The historical log-determinant curvature proxy around -5.63 is insensitive to the position-independent metric factor at fixed coordinates. Coordinates themselves may change during training, and the actual conformal metric need not be flat. This proxy is not a general sectional-curvature estimator. The historical branch is distinct from the current public branch.

The “6/6” historical audit is **not a certification**. Its record lacks an immutable source/checkpoint binding. The inspected loader accepts missing weights (random initialization) and partial state loads; this is a provenance risk, not proof that this run used random weights. Its “transport” criterion measures coordinate displacement, and its “energy” criterion measures relative change in squared coordinate norm, not the Hamiltonian. That relative change is a fraction despite the `max_drift_pct` field name. The triangle check clamps norms without projecting coordinates. Entropy and distance from uniform do not establish useful specialization or component diversity.

The [inspected audit source snapshot](benchmark-results/historical-audit-source.py.txt) is published for review, not asserted to be the immutable source of the March run. The new ARC parser scores the saved historical first-pair outputs 0/100: the previously accepted answer contains a correct grid followed by a code-array expression, rejected by the new last-grid-candidate rule. This is a parser-policy sensitivity result, not a new inference result or a claim that the model emitted no correct answer.

## Later comparisons

The [expanded export/image audit](export-image-audit.md) also withdraws empirical interpretations of the synthetic thesis loss, affinity and spectrum plots, the image's “50–75% step reduction,” and the negative-curvature label on a downward paraboloid. Public synthetic PNGs now carry visible labels. Imposed topology layouts and the inspected standard-Qwen2 GGUF directories do not establish learned hyperbolicity or hypercomplex execution. The two inspected GGUFs are separate from the October cp300 benchmark artifacts.

The 20 September 2026 paired token-NLL probe found lower NLL for a vanilla-LoRA control than for the geometric arm on all 24 paired examples. Its small sample, unequal parameter counts, and different layer coverage prevent causal attribution to geometry.

The 27 October 2026 local multi-benchmark run found no statistically significant individual paired task differences. Its custom weighted index was 61.1 for the base, 60.0 for the exported cp300 checkpoint, and 64.7 for a reconstructed adapter composite. The cp300 export did not include the runtime geometric adapter; the reconstructed arm grafted a surviving base-trained adapter onto cp300 and used a different loading/quantization path. This exploratory reconstruction is not an independent or parameter-matched confirmation of benefit.

The 252 prompt-free paired recorded grades now reproduce the October scores and exact McNemar tests offline. This verifies arithmetic, not grader correctness. Failure to reject a difference is not equivalence, “no harm,” or preservation of capabilities. The September published rows also reproduce a two-sided sign test (24/24 control wins, p = 1.1921e-7); this is distinct from the historical t-test.

See the exact aggregate artifacts:

- [October per-benchmark results](benchmark-results/aa-index-2026-10-27.csv)
- [September paired token-NLL data](benchmark-results/matched-token-nll-2026-09-20.csv)
- [Fixed-operator audit](benchmark-results/faithfulness-audit-A.json)
- [Historical ARC-Challenge adapter results](benchmark-results/arc-challenge-adapter-2026-01.json)
- [Historical ARC-Challenge base results](benchmark-results/arc-challenge-base-2026-01.json)
- [Historical ARC-AGI aggregate](benchmark-results/arc-agi-summary.json)
- [October paired recorded grades](benchmark-results/aa-paired-outcomes-2026-10-27.csv)
- [Historical first-pair strict rescoring](benchmark-results/arc-agi-rescore.json)
- [Provenance and audit-method limitations](benchmark-results/provenance.json)
- [Offline verification output](benchmark-results/verification.json)

Current repairs cover radial Poincare projection, unbiased distance, identity initialization, and shared exact ARC scoring/accounting. These change historical behavior and do not retroactively improve any checkpoint. The README explicitly bounds the tested core; the separate legacy learned-connection and training paths are not certified. Older PDFs and manuscript sources remain archived and should be read with this correction.
