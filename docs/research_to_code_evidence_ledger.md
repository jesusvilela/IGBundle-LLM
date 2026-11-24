# Research-to-code evidence ledger

Research documents generate bounded hypotheses; code plus tests establish local
operators; paired experiments establish empirical effects. None substitutes for
another.

| Source | Claim | Decision and binding | Status |
|---|---|---|---|
| IGBundle thesis | A fiber-bundle adapter can improve reasoning | Retain as falsifiable hypothesis; require matched base/control/geometric runs | Open |
| `AUTORESEARCH/thesis.md` | Geometry, information geometry, NMEM, GSP and falsification form research axes | Split into independent operator claims and tests | Partially lowered |
| AUTORESEARCH run `20260328_001617` | Produce a recent fiber-bundle/transformer synthesis | Keep its plan; completion requires source matrix, citations and verification outputs | Planning complete; synthesis absent |
| July audit/falsification (`579dd7c`) | Default fixed-conformal curvature is weight-invariant | Do not report it as learned geometry; require weight sensitivity at fixed coordinates | Accepted/tested locally |
| Claim-safe benchmark policy | Task claims require exact provenance and paired raw runs | Adopt as publication gate via `benchmark_unified_smoke.py` | Accepted; run blocked in model load |
| GRD repository + merged PR #1, refreshed 2026-09-20 | GRD-0...4 now have a NumPy reference implementation over exact sphere/hyperboloid manifold interfaces | Adopt the coherent `proj/exp/log/transport` interface and matched RSGD controls as design evidence. Do not port the optimizer wholesale into adapter training | E1-E4 reproduced locally; E5 failed |
| GRD local E5 rerun | A loss-based acceptance certificate prevents harmful overshoot | Local run rejected 198/200 steps but finished worse than the uncertified run (-0.2971 vs -1.2562); checked-in positive result did not reproduce | Certificate claim rejected for this environment/config; redesign required |
| GRD spectral/holonomy/telos terms | Global connection structure improves optimization | Updated implementation still leaves these terms unexercised; smooth-manifold tests do not carry graph/connection cycles | Open; no transfer claim |
| Causal training audit | Fisher refresh erased gradients; path-prefixed losses got zero weights; HF path silently fell back | Preserve gradients, canonicalize keys, fail closed | Implemented; causal tests pass |
| Lambda/sheaf audit | Abstraction dimension was invalid; terms are neural surrogates, not law-level structures | Fix abstraction; emit `section_representation_alignment` and `local_section_consistency`; keep mathematical laws open | Partially implemented |
| Claim-safe reconciliation | Active repairs must survive the smaller auditable branch without importing the lab tree wholesale | Port gradient-preserving Fisher refresh, explicit loss registry, fail-closed state/adapter gates, conformal Levi-Civita connection, witnessed holonomy, and bounded offline smoke path | Implemented; claim-safe full suite 12/12 passes |
| September 20, 2026 paired evaluation of March checkpoints | Geometric A should beat or distinguish itself from vanilla-LoRA B under shared training conditions | Unsloth evaluation on 24 identical held-out examples (64 tokens each): base 2.3188 NLL, A 1.9352, B 0.8187; B wins 24/24; mean A-B +1.1165 (95% paired t CI [0.9144, 1.3186]), p=5.83e-11. [Per-example scores](benchmark-results/matched-token-nll-2026-09-20.csv) | No geometric advantage observed in these arms; parameter-count and layer-coverage confounds prevent isolating geometry |
| October 27, 2026 local subset evaluation | Determine whether the exported cp300 model or reconstructed adapter composite differs from base | Fixed subsets (n=8–98), temperature 0, same prompts/subsets: custom index base 61.1, merged cp300 60.0, reconstructed composite 64.7. No individual paired comparison significant; corrected-vs-cp300 McNemar p≥0.63. [Per-benchmark summary](benchmark-results/aa-index-2026-10-27.csv) | Exploratory only. Original cp300 export did not include the runtime geometric adapter; reconstructed composite grafted a base-trained adapter onto cp300 and used a different quantization path. Small samples and post-hoc reconstruction do not establish efficacy |

## Lowering schema

```text
source -> bounded claim -> evidence class -> accept/reject/defer
       -> operator/config -> invariant -> counterexample -> test
       -> experiment manifest -> result + remainder
```

A completed plan is `planned`, not completed research. GRD experiments start at
GRD-0 with a defined metric and exponential map and compare against matched
natural-gradient/Riemannian baselines before modal extensions. The current
IGBundle lift follows that boundary: its conformal metric now induces an
analytic Levi-Civita connection, matrix-free transport, geodesic integration,
and a closed-loop holonomy witness. This establishes executable geometry, not
an empirical language-model advantage.

The paired probe does not validate the old `parameter-matched` label: A
recorded 4,160,713 trainable parameters and B 585,728. It shows a directional
difference for these executed arms: the vanilla control had lower NLL on every
paired held-out sample. This does not isolate a geometric effect. The next
valid experiment must equalize trainable parameters and layer placement while
ablating only the geometric objective/operator.

Sources: https://github.com/jesusvilela/Geodesic-Resonance-Descent and
https://github.com/jesusvilela/Geodesic-Resonance-Descent/pull/1.
