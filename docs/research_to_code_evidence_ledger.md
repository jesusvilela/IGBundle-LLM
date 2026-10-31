# Research-to-code evidence ledger

Research documents generate hypotheses; implementation and operator tests establish local behavior; matched task experiments are required to establish empirical effects. None substitutes for another.

| Claim | Evidence | Conclusion |
|---|---|---|
| Fiber-structured geometric adapters improve reasoning | The October 2026 local multi-benchmark comparison found no significant individual paired task differences; its custom index is exploratory. The September 2026 token-NLL probe favored its vanilla-LoRA control on all 24 paired examples. | Open hypothesis; no general advantage established. |
| The October reconstructed composite improved the model | Its custom index was 64.7 vs 60.0 for cp300 and 61.1 for the base; recomputed exact paired McNemar p-values vs cp300 are 0.625–1.00. The composite was assembled post hoc from an adapter trained on the base, not the original cp300 companion; model loading/quantization also differed. | Descriptive point estimates only; not causal or independent confirmation. |
| The exported cp300 model executes the geometric adapter | The October audit reports the merged checkpoint's served tensor index contains Qwen tensors only; the geometric adapter was a separate runtime module and was absent from the exported model. | Do not describe the merged cp300 evaluation as an inference-time geometric-adapter evaluation. |
| Training learned curvature near -5.63 | At fixed coordinates, the historical spatial log-determinant proxy cancels its position-independent metric factor. Coordinates may still change, and the actual metric need not be flat. | Retracted as learned-curvature evidence; not a statement about every estimator or the current branch. |
| The “6/6” audit certifies geometric faithfulness | The record lacks immutable run/source/weights binding. The inspected loader allows random initialization and partial loading; transport and energy are proxies, not law-level tests. | Historical report only; not certified operator geometry. |
| Historical ARC results show a gain | ARC-Challenge zero-shot `acc_norm` was 54.86% for both adapter and base on 1,172 items each; the ARC-AGI legacy scorer reported 1/100 using only first test pairs and substring matching. | No gain demonstrated; no all-test-pair solve claim. |
| Historical natural-gradient/geometry terms reduce training or inference cost | No suitable matched, reproducible comparison is available in the public evidence artifacts. | Withdrawn; do not report efficiency gains or overheads as established results. |
| The current default Poincare affinity is numerically checked | Analytic origin-distance, exact diagonal, symmetry, sampled triangle, gradient, and mixed-precision small-adapter regressions. | Bounded unit-tested implementation, not a trained-model result. |
| The October statistics can be recomputed | 252 paired recorded grades reproduce custom indices and exact McNemar tests with a standard-library script. | Arithmetic reproduced; historical graders and provenance remain limited. |
| The ARC runners measure exact task success | Shared parser, all test pairs, deterministic selection and failure accounting are covered with synthetic generators. | Scoring protocol checked; no model/backend inference rerun. |
| Surviving GGUF exports execute hyperbolic/hypercomplex operations | Two inspected directories have standard Qwen2 tensor names; the packaged fast path omits its separate adapter. Prefix hashes and full tensor descriptors are published. | No such runtime integration established; effects of training in ordinary weights are not ruled out. |
| Thesis training/affinity/SVD images are measurements | Inspected generator uses exponentials plus noise, random block affinities and power-law spectra. | Synthetic illustrations; the public PNGs now have embedded warnings. |
| The archived downward paraboloid illustrates negative curvature | Its induced Gaussian curvature is analytically positive; the assigned negative heatmap is unrelated. | Incorrect label, corrected by an independent analytic reference. |
| Hyperbolic and hypercomplex are interchangeable | A metric and an algebra product are different structures; inspected source-name scans and runtime trace establish no hypercomplex integration. | Do not transfer claims between structures or from separate repositories. |
| Length-preserving scaling certifies hyperbolic parallel transport | The archived scale-only map telescopes to identity on every closed loop; a prescribed K=-1 triangle requires nonzero rotation. Independent gyration, Lorentz transport and angle-defect checks agree. | Scaling alone is insufficient. The new opt-in transport reference is not a model or integrator certification. |
| The legacy adapter's forward path performs parallel transport | Its intended coordinate blending formula needs no metric or connection. An unused rank-four metric construction caused multi-token failure; the repaired path passes tiny execution checks without calling either transport or connection methods. | Execution repaired, not geometric laws certified. Chunked smoothing needs previous-coordinate history. |
| A GGUF directory fingerprint identifies the executed weights | Negating only the output weight in a tiny native-executed Qwen2 leaves its directory report unchanged except the filename, but changes logits by 0.77686. Complete file hashes differ. | Prefix fingerprints do not bind tensor values or model behavior. This counterexample does not load the archived exports. |
| The ARC companions fit the surviving 0.5B GGUF | Non-executing metadata inspection finds adapter input/output width 3584, versus GGUF width 896. The separate projector's 896-wide output matches the adapter bottleneck, not its external hidden-state interface. | Direct insertion is incompatible. Shapes do not establish a replacement bridge, vision alignment, exact restoration or learned geometry. |

## Published data

**Archived companion metadata:** four named files yield 226 descriptors with
52,399 metadata-read bytes per complete pass and no tensor-storage member opens.
The ARC width mismatch is explicit; dist/varied core widths alone do not certify
restoration. Under the inspected packaged defaults, the dist runner disables
modules corresponding to 22 saved vision/geodesic-attention keys and uses
non-strict loading. [Metadata and source excerpts](benchmark-results/companion-state-metadata.json) ·
[Reader controls and compatibility limits](export-image-audit.md#archived-companion-metadata-checkpoint).
Tiny genuine PyTorch saves check the restricted decoder; archived payloads,
storage checksums, trained-state provenance and geometric laws are not checked.

**Tiny native GGUF execution:** synthetic 17,136/35,296-parameter F32 Qwen2 files
run in the pinned native library, with tied/untied outputs and grouped-query
attention. Full/incremental logits match independent PyTorch results within
declared tolerance (largest observed absolute error `1.79e-7`); malformed Q/K
shapes are rejected. [Recorded results](benchmark-results/native-gguf-reference.json)
and [runtime scope](export-image-audit.md#tiny-native-gguf-checkpoint).
This is not archived-weight, converter, tokenizer, quantized-kernel or
geometric/hypercomplex integration certification.

**Legacy adapter execution:** six bounded CPU tests cover the repaired
`GeometricIGBundleAdapter` forward, nonzero gradients/output, causal prefixes and
a strict tensor-only module-state roundtrip. The preserved smoothing formula is
not parallel transport and uses no connection.
[Reproduced failure, hosted result and limits](export-image-audit.md#legacy-adapter-execution-checkpoint).
This is not historical restoration, default-loader coverage, cached generation,
auxiliary-loss or training-pipeline certification.

**Qwen2 layout contract:** the two inspected GGUF directories pass 339/339 and
290/290 shape checks against the pinned native dense-scalar profile. Optional
biases and output tying are explicit; malformed Qwen2 output is rejected by the
exporter and ambiguous Q8_1 extents are not certified.
[Profile, source revision and exclusions](export-image-audit.md#qwen2-directory-shape-contract).
This adds structural evidence, not tensor-value, native-loader or geometric
runtime certification; prefix hashes still exclude all weights.

**Hyperbolic transport:** a small opt-in reference provides full gyration and
dual cotangent transport. The fixed loop rotates by -0.238857852 radians,
where the archived scale-only formula returns zero.
[Numerical witness](benchmark-results/transport-reference.json).
The metric is prescribed and no adapter/GGUF path uses this reference.
The archived momentum caller is not repaired or certified by these tests.

**Declared-LoRA coverage:** six public load/merge paths verify exact serialized
keys and restored values. Tiny real PEFT checks cover nonzero wrapped-save state
when the wrapper structure matches, and reject mismatches. This is post-load
verification without rollback, not historical checkpoint certification.

**Tiny real backend:** random Qwen2 plus PEFT and the default adapter is now
exercised on hosted CPU, including restoration, gradients and cached generation.
Versions and scope are pinned. The joint LoRA/IGBundle logit roundtrip saves PEFT
before wrapper insertion; the separate LoRA coverage check also covers wrapped
saves with matching structure.
This narrows the integration gap, not the historical provenance or efficacy gap.

**Exact algebra reference:** finite coefficient certificates for an explicitly
defined Cayley-Dickson product distinguish quaternion/octonion laws from vector
dimension or Hamiltonian terminology.
[Certificate](benchmark-results/algebra-reference.json).
This independent reference is not wired into an adapter or GGUF runtime.

**Export preflight:** missing converter/config/weights and output collisions are
rejected before model allocation; LoRA-only scope is explicit. Dependency-free
contracts exercise dry runs and fake backend orchestration, not a real merge.
The GGUF directory inspector checks conversion output structure, not model values.

**Projection-display correction:** the standalone renderer now plots a bounded
deterministic prefix of one selected projection, not an inferred manifold.
Exact key selection, source-file hashing and explicit failure replace random
scaffolds and filename-derived geometry labels.
[Contract and limits](export-image-audit.md#projection-display-checkpoint).

**Demo correction:** synthetic training trajectories and literal ablation/statistics
fallbacks were removed. Live telemetry now labels Gaussian dispersion and categorical
probabilities explicitly; missing data is unavailable. Headless checks cover tiny-array
rendering and aggregation, plus real Gradio UI construction with a stubbed model and
intercepted launch. Server operation and model/backend integration remain unverified.
[Display contract](export-image-audit.md#demo-evidence-checkpoint).

**Runtime restoration checkpoint:** the default adapter now has a fail-closed
tensor-coverage contract, used by five public HF loaders. Tiny synthetic tests
cover restoration and actual wrapped execution, not real HF/PEFT compatibility.
Historical checkpoint identity and the separate GGUF execution gap remain
unresolved. [Contract and exclusions](export-image-audit.md#default-adapter-restoration-checkpoint).

- [October 27, 2026 per-benchmark summary](benchmark-results/aa-index-2026-10-27.csv)
- [September 20, 2026 paired token-NLL rows](benchmark-results/matched-token-nll-2026-09-20.csv)
- [Faithfulness audit summary](benchmark-results/faithfulness-audit-A.json)
- [Historical ARC-Challenge adapter](benchmark-results/arc-challenge-adapter-2026-01.json) and [base](benchmark-results/arc-challenge-base-2026-01.json)
- [Historical ARC-AGI summary](benchmark-results/arc-agi-summary.json)
- [October paired recorded grades](benchmark-results/aa-paired-outcomes-2026-10-27.csv)
- [Strict historical first-pair rescoring](benchmark-results/arc-agi-rescore.json)
- [Provenance manifest](benchmark-results/provenance.json)
- [Offline verification](benchmark-results/verification.json)
- [Expanded export/image audit](export-image-audit.md)
- [GGUF cp600-named directory](benchmark-results/gguf-dist-directory.json) and [0.5B directory](benchmark-results/gguf-arc-directory.json)
- [Image/source fingerprints and excerpts](benchmark-results/artifact-audit.json)

The October and September raw prompt/generation logs are not included here. The published files contain aggregate scores or prompt-free paired measurements and recorded grades. Source hashes do not replace absent artifacts. The September historical source commit is local-only; its dataset revision and complete loader are not bundled. Reproduce claims only within the stated limitations; this is not full end-to-end reproducibility from a fresh clone.
