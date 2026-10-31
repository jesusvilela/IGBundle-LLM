# IGBundle: Fiber-Structured Geometric Adapters for Language Models

**A research prototype exploring whether information-geometric operations and fiber-structured adapters can provide a useful inductive bias for language models.**

[![License: All Rights Reserved](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)
![Status: Experimental Research](https://img.shields.io/badge/Status-Experimental%20Research-purple.svg)

> **Evidence status (31 October 2026):** The geometric inductive-bias idea remains an open research hypothesis, but the available experiments do **not** establish a general task-performance advantage, learned hyperbolicity, or training/inference efficiency gains. Previously advertised accuracy, “MFR,” ablation, and efficiency claims are retracted. Recorded-result arithmetic is reproducible offline; historical model runs and graders have not been independently reproduced. See the [correction notice](docs/correction-notice.md), [evidence ledger](docs/research_to_code_evidence_ledger.md), and [verification output](docs/benchmark-results/verification.json).

## Research design

IGBundle explores Poincare-ball coordinates, categorical fiber components, and geometric or dynamical operators as a possible structure for adapting language models. These describe the intended parameterization; they are not evidence that a trained model has learned semantic hierarchy, a fiber bundle, sheaf structure, or hyperbolic geometry.

In a historical experimental branch, a spatial log-determinant proxy cancels a position-independent metric factor at **fixed coordinates**. That is not a measurement of learned curvature. Coordinates may change with training, and the actual metric need not be flat. This historical proxy is not the current public branch's default adapter. Operator checks, geometry diagnostics, and downstream task evaluation answer different questions.

## What the evidence currently shows

### October 2026 local multi-benchmark evaluation

A 27 October local run compared an exported Qwen2.5-7B checkpoint (cp300), the base model, and a reconstructed composite with runtime IGBundle adapters. It used greedy decoding on fixed, small subsets (8–98 examples per task). Its “intelligence index” is a custom weighted average, not a standardized benchmark:

| Variant | Custom weighted index | Scope |
|:---|---:|:---|
| Base Qwen2.5-7B | 61.1 | Q4_K_M GGUF |
| Merged cp300 checkpoint | 60.0 | Q4_K_M GGUF |
| Reconstructed adapter composite | 64.7 | Separate NF4/bf16 loading path |

No individual paired task comparison was statistically significant. The reconstructed composite had higher point estimates than cp300 on 5/6 tasks, but every recomputed exact McNemar p-value was at least 0.625 (0.63 in the rounded historical table). These small subsets cannot establish equivalence or a broad benefit.

**Checkpoint caveat:** the exported cp300 weights did not contain the runtime geometric adapter. The reconstructed arm grafted a surviving adapter trained on the base model onto cp300; it was not the original trained checkpoint. Model assembly and quantization also differed. This exploratory, post-hoc reconstruction does not isolate a geometric effect. The [per-task summary](docs/benchmark-results/aa-index-2026-10-27.csv) includes sample counts, scores, weights, parse failures, and paired p-values.

The [252 paired recorded grades](docs/benchmark-results/aa-paired-outcomes-2026-10-27.csv) now permit recomputing accuracies, custom indices, and exact two-sided McNemar tests. These are recorded grader decisions, not newly verified answers. Source hashes and missing provenance are disclosed in the [provenance manifest](docs/benchmark-results/provenance.json). Nonsignificance does not demonstrate equivalence or “no harm.”

| Task | n | Base | Merged cp300 | Reconstructed composite | McNemar p, composite vs cp300 |
|:---|---:|---:|---:|---:|---:|
| MMLU-Pro | 98 | 53.1% | 48.0% | 51.0% | 0.68 |
| ARC-Challenge | 60 | 90.0% | 88.3% | 91.7% | 0.69 |
| MATH-500 | 40 | 40.0% | 45.0% | 50.0% | 0.73 |
| GSM8K | 30 | 83.3% | 76.7% | 80.0% | 1.00 |
| IFBench-lite | 16 | 68.8% | 68.8% | 81.2% | 0.63 |
| Degeneration probes | 8 | 75.0% | 75.0% | 75.0% | 1.00 |

### September 2026 paired token-NLL probe

On 24 examples reported as held out, with 64 tokens scored per example, mean token negative log-likelihood was 2.3188 for the frozen base, 1.9352 for geometric adapter A, and 0.8187 for vanilla-LoRA control B. B had lower NLL on all 24 examples. The mean paired difference `A − B` was +1.1165 (95% paired t interval `[0.9144, 1.3186]`). The independently recomputable two-sided sign-test p-value is `1.1921 × 10⁻⁷`; it is not the historical paired t-test p-value.

This is a small language-modeling probe, not task accuracy. A had 4,160,713 trainable parameters and one geometric layer; B had 585,728 parameters and 13 LoRA layers. The comparison does not isolate geometry, layer coverage, or objective. [Per-example scores](docs/benchmark-results/matched-token-nll-2026-09-20.csv) are published without prompt text.

### Historical diagnostics and task evaluations

- The historical audit file reports **6/6 checks passed**, but does not bind the run to immutable source and checkpoint hashes. The inspected audit loader allows random initialization when weights are missing and non-strict loading. Its displacement and squared-coordinate-norm proxies do not verify parallel transport or Hamiltonian energy conservation. Retained as a historical report, **not operator certification**. [Audit data](docs/benchmark-results/faithfulness-audit-A.json) · [method limitations](docs/benchmark-results/provenance.json).
- Historical zero-shot ARC-Challenge `acc_norm` was **54.86% for both** an adapter checkpoint and the base (1,172 test items per run). This is a separate protocol and dataset size from the October subset run. [Adapter results](docs/benchmark-results/arc-challenge-adapter-2026-01.json) · [Base results](docs/benchmark-results/arc-challenge-base-2026-01.json).
- A separate historical ARC-AGI evaluator reported **1/100 (1%)**, but scored only the first test pair per task and allowed substring matching. This is not an all-test-pair exact-task solve rate. Model-First Reasoning format compliance was **0/100**; it is a prompt-format metric, not geometric faithfulness. [Historical aggregate](docs/benchmark-results/arc-agi-summary.json) · [strict rescoring of the saved first-pair answers](docs/benchmark-results/arc-agi-rescore.json).

The new last-grid-candidate parser scores those saved first-pair outputs **0/100**. The previously accepted answer contains the correct grid followed by a code-array expression, which the new policy rejects. This is a **parser-policy sensitivity result**, not new inference or proof that the model never emitted a correct answer. The repaired scorer is a declared local protocol, not an official ARC leaderboard evaluator.

The historical value near -5.63 is not evidence of learned hyperbolicity. For `g(x) = lambda(x) G`, `log det g = D log lambda + log det G`; spatial differences remove the second term when G is position-independent. This limited algebraic statement does not certify the other claims in the historical [falsification draft, section 2.6](https://github.com/jesusvilela/IGBundle-LLM/blob/579dd7c2220abc146bf7380afb55f188f6f461b1/draft_paper_falsification.md).

## Verified implementation boundary

| Surface | What is checked | What is not established |
|---|---|---|
| Default-adapter restoration | Exact registered tensor coverage, shapes, finite values and safe dtype conversion before copying; missing-weight rejection in five HF entry points; tiny restored-operator execution | Historical checkpoint compatibility, base/LoRA identity, production HF loading, or GGUF adapter execution |
| Default `IGBundleAdapter` / Poincare affinity | Radial projection, analytic origin distances, exact zero diagonal, symmetry, sampled triangle inequalities, finite/checked gradients, float32/bfloat16 small-tensor forward/backward, identity initialization | Trained-model quality, learned geometry, or all possible numerical inputs |
| `arc_eval.py`, HF and GGUF ARC runners | Shared deterministic selection, strict grid equality, all test pairs, errors retained and run invalidated, explicit empty-run rejection, output overwrite protection | Backend model loading or restoration of separate runtime adapter weights; no new model evaluation was run |
| Historical evidence | Recomputed NLL means and paired statistics; October accuracies, indices and McNemar tests; source hashes and first-pair ARC rescoring | Independent grading, dataset non-contamination, or complete reproduction from a fresh clone |
| Synthetic F32 Qwen2 GGUFs | Native full/incremental logits compared with PyTorch, tied/untied output weights, grouped-query attention and malformed-shape rejection | Not the archived exports, tokenizer semantics, quantized kernels, public converter or geometric/hypercomplex integration |
| Legacy `GeometricIGBundleAdapter` forward | Tiny identity/nonzero execution, gradients, causal prefixes and strict module-state roundtrip; repaired multi-token coordinate smoothing | No connection is used by this forward path. Not parallel transport, cached generation, historical checkpoint restoration or a supported training pipeline. |
| Legacy `geometry/riemannian.py`, training and experimental geometry modules | **Outside the validated geometric core** | The metric is position-independent while connection coefficients come from an unrelated network; these do not establish Levi-Civita geometry, categorical laws, or sheaf laws. The legacy `train.py` entry point also contains an incomplete trainer block and is not a supported training command. |

The Poincare affinity correction changes historical numerics; old checkpoint scores must not be attributed to the corrected implementation without reevaluation. The default adapter is now exactly residual-identity at initialization (both output weights and bias are zero). No new checkpoint or performance result is implied.

## Expanded archive audit: GGUFs and images

Two surviving model GGUFs were inspected without reading tensor payloads: the cp600-named Q8_0 export contains **339 ordinary Qwen2 tensors**; the 0.5B F16 export contains **290**. Their packaged fast path does not load the separate geometric adapter. These files are distinct from the October cp300 benchmark artifacts. Prefix fingerprints identify the inspected headers, **not the full checkpoints**.

The directory audit now checks **339/339 and 290/290 tensor shapes**, not only
names and byte extents, against a pinned native llama.cpp dense-Qwen2 profile.
Grouped-query attention, feed-forward and vocabulary dimensions agree.
Optional biases and tied output embeddings are handled explicitly.
The exporter rejects failed Qwen2 layout checks; ambiguous Q8_1 extents are
rejected rather than guessed. This is not native-loader, tokenizer, tensor-value
or inference certification. [Profile and source revision](docs/export-image-audit.md#qwen2-directory-shape-contract).

Nine research images were reviewed against their generators. The thesis loss, affinity and spectrum plots are synthetic and now labeled inside the public PNGs. An archived “negative curvature” drawing actually depicts a positive-curvature paraboloid; prescribed layouts and simulated convergence curves are not learned-geometry results.

![Analytic geometry reference, not a model measurement](docs/assets/geometry-reference.svg)

**Hyperbolic geometry is not hypercomplex algebra.** A verified Poincare operator, a high-dimensional vector, or a Hamiltonian module does not establish a quaternion/octonion product or its execution in a GGUF. No such integration is established by the inspected paths. The geometric approach remains a hypothesis, not a claim rescued by relabeling diagrams.

[Full export/image audit and derivations](docs/export-image-audit.md) · [GGUF cp600-named directory](docs/benchmark-results/gguf-dist-directory.json) · [GGUF 0.5B directory](docs/benchmark-results/gguf-arc-directory.json) · [Image/source fingerprints and excerpts](docs/benchmark-results/artifact-audit.json).

## Project structure

```
src/igbundle/
├── geometry/          # Legacy experimental geometry; not certified
├── modules/           # Default mixture adapter, KL, affinity, and losses
├── integrations/      # Hugging Face block wrapping
└── training/          # Legacy experimental training utilities

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

### Low-resource verification

With any Python 3.10+ interpreter, no third-party packages are needed for:

```bash
python scripts/verify_evidence.py
python -m unittest discover -s tests -p test_evidence.py
python -m unittest discover -s tests -p test_arc_eval.py
python -m unittest discover -s tests -p test_gguf_inspection.py
python -m unittest discover -s tests -p test_artifact_audit.py
```

In an existing environment with PyTorch, the bounded operator checks are:

```bash
python -m unittest discover -s tests -p test_geometry.py
python -m unittest discover -s tests -p test_ig.py
python -m unittest discover -b -s tests -p test_checkpoint.py
python -m unittest discover -b -s tests -p test_legacy_geometric_adapter.py
```

Use `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` to bound local resources. None of these commands loads a model, downloads data, or runs LLM training. Do not use blanket test discovery as a lightweight check: the legacy `test_pipeline.py` invokes the unsupported model-training entry point.

The [CPU evidence workflow](.github/workflows/evidence.yml) runs bounded arithmetic,
operator, tiny HF and synthetic native-GGUF checks. The latter also compiles the
pinned native dependency with two hosted CPU workers; it is not a lightweight
local command. A green result means these contracts passed, not that the whole
experimental repository or its scientific claims are certified.

### Fail-closed adapter restoration

`app.py`, `benchmark.py`, `bench_arena.py`, and `validate_effects.py` now require
`adapter_weights.pt` when an adapter checkpoint is requested; `evaluate.py`
requires `full_state_dict.pt`. Missing or incompatible state no longer silently
leaves random default adapters active. The shared loader uses CPU,
`weights_only=True`, and exact registered names; it does not guess key prefixes.
Declared PEFT load errors propagate instead of becoming warnings.

This is an intentional compatibility break for partial checkpoints. It is not
a converter for the archived `adapter_refined.pt` or `arc_adapter.pt`, and does
not repair the separate GGUF runner. Full-state mode restores only default
adapter tensors, not the base or LoRA. See the
[restoration contract and limits](docs/export-image-audit.md#default-adapter-restoration-checkpoint).

### Demo evidence boundaries

The experimental `app.py` no longer substitutes synthetic training curves,
hard-coded ablation bars, or invented statistics when evidence is unavailable.
Those panels explicitly show missing evidence. The withdrawn performance and
efficiency claims have been removed from the demo itself.

Live sigma values are **Gaussian standard deviations, not curvature**.
The categorical heatmap shows probabilities, not pairwise affinities. Entropy is
descriptive, and the old sigma interval counter is no longer called “MFR” or
graded as geometric quality. Missing telemetry is unavailable, not zero.
Accumulation supports varying decode lengths and retains at most 4,096 sigma
samples plus fixed-size summaries; the histogram is explicitly a prefix sample.

Plotting checks use tiny arrays, not an LLM:

```bash
python -m unittest discover -b -s tests -p test_demo_evidence.py
```

They require NumPy, Matplotlib, Pillow and Gradio; the hosted workflow runs them headlessly.
`requirements-demo.txt` lists optional UI dependencies in addition to the base
requirements. The launcher now requires an explicit `--checkpoint` path.
The UI is constructed with a stubbed model and intercepted server launch.
These checks do **not** validate a running server or real model/backend loading.
See the [demo correction scope](docs/export-image-audit.md#demo-evidence-checkpoint).

### Parameter display, not topology

The retained `generate_braintop_viz.py` entry point now produces a deterministic
HTML/SVG heatmap of stored input-projection weights, not a purported manifold.
It rejects missing weights, ambiguous tensor selection and malformed values.
No random scaffold, filename-derived curvature label or semantic node metadata
is used.

```powershell
python generate_braintop_viz.py --checkpoint "<trusted-checkpoint-directory>" --tensor-key "<exact.input_proj.weight>" --output output\parameters.html --lite
```

The key may be omitted only when exactly one input projection exists.
Input is limited to 64 MiB and the selected matrix to 4,194,304 values; the display
contains only the first 32 x 32 values in lite mode, otherwise at most 64 x 64.
The source hash, shape, dtype, prefix extent and color scale are embedded.
Only trusted local tensor checkpoints are supported; `weights_only=True` is not
a general resource-isolation guarantee. This does not restore an adapter or
certify learned geometry. See the [display contract](docs/export-image-audit.md#projection-display-checkpoint).

### GGUF export preflight

`export_gguf.py` requires an explicit converter and `--lora-only` acknowledgement:
it merges PEFT LoRA weights, **not nonlinear runtime geometry**. Use `--dry-run`
to check paths and configuration without importing model libraries or allocating
weights:

```powershell
python export_gguf.py --base "<exact base identifier from adapter_config.json>" --checkpoint "<trusted PEFT directory>" --converter "<trusted converter.py>" --output output\new_export --lora-only --dry-run
```

Execution additionally requires the model dependencies and substantial CPU RAM/disk.
It reserves fresh `<output>_hf` and `<output>.gguf` destinations; failures retain
partial files and a retry must use a fresh prefix. The converter must support
the HF-directory, `--outfile`, and `--outtype f16` interface and writing the
reserved GGUF destination. Local preflight and mocked orchestration are checked;
real model merge, converter compatibility and inference from this exporter remain
unverified. The separate synthetic native check below bypasses this exporter.
For declared Qwen2 output, the bounded directory name/shape profile must also
pass. Other architectures receive generic extent checks, not this shape profile.

### Exact hypercomplex reference

`scripts\algebra_reference.py` supplies an independent Cayley-Dickson reference
in real dimensions 1, 2, 4 and 8. Exact integer coefficient checks establish
norm composition, conjugation reversal and alternativity for that product;
associativity holds through dimension 4 and fails in dimension 8.
Under the declared basis convention, `(e1*e2)*e4 = e7` but
`e1*(e2*e4) = -e7`.

Run `python scripts\algebra_reference.py` with no third-party dependencies.
The [certificate](docs/benchmark-results/algebra-reference.json) is a mathematical
reference, **not an adapter integration, learned geometry or model result**.
It supplies no hypercomplex hyperbolic metric or GGUF implementation.

### Tiny real-backend integration

Hosted CPU checks now exercise a randomly initialized Qwen2 with two layers,
hidden width 16 and fewer than 100,000 base parameters, using PyTorch 2.6.0,
Transformers 4.48.3, PEFT 0.14.0 and Accelerate 1.2.1.
They check identity wrapping, real PEFT save/reload before wrapper insertion,
strict adapter restoration, gradients and cached generation.

Nonzero restored adapter values affect logits; restoration leaves base and LoRA
tensors unchanged. Cached and full-prefix logits agree within declared tolerance.
No pretrained model or dataset is downloaded. These are execution contracts,
not accuracy, training convergence, historical checkpoint or GGUF certification.
The full app/server and quantized backends remain outside this check.

Pinned optional dependencies are in `requirements-test-hf.txt`.
Run `python -m unittest discover -b -s tests -p test_tiny_hf_integration.py`
in a compatible CPU environment.

### Declared LoRA restoration

The six public HF load/merge paths now check exact default-LoRA tensor keys,
shapes, finite values and restored values. Missing or extra tensors and wrapper
prefix mismatches fail explicitly; no names are guessed or rewritten.
The optional Unsloth path receives the same post-load check, but was not executed.

Tiny pinned PEFT checks cover both unwrapped saves and saves made after IGBundle
wrapper insertion. The latter require the same wrapper structure when loading.
Only one `default` LoRA adapter is supported by this contract.
This is a post-load check, **not rollback**: a failed caller must discard its
partially constructed model. Coverage does not establish base identity, training
history, historical compatibility, or full application correctness.

### Hyperbolic transport reference

The opt-in `igbundle.geometry.poincare.PoincareBall` supplies geodesic tangent
and cotangent transport for the **prescribed** metric
`g_x = [2 / (1 - c ||x||^2)]^2 I`, with fixed `c > 0`.
It includes gyration, the rotation missing from the archive's scale-only method.
Momentum uses the dual map, not the tangent map.

For the geodesic triangle `(0,0) -> (.3,0) -> (0,.4) -> (0,0)` at `c=1`,
the returned tangent rotates by `-0.238857852` radians, matching the negative
angle defect; scale-only transport returns zero rotation.
Independent hyperboloid comparisons cover dimensions 2/3/8 and `c=.25/1/4`,
alongside metric products, inverse transport, gradients and explicit domain errors.
[Numerical witness](docs/benchmark-results/transport-reference.json) and
[audit, derivation and limits](docs/export-image-audit.md#hyperbolic-transport-checkpoint).

Run `python scripts\transport_reference.py` in a compatible PyTorch environment.
This is a transport-only reference, not a drop-in dynamics replacement.
It is **not wired into either adapter or GGUF**, does not repair the archived
Hamiltonian integrator, and establishes no learned curvature or model benefit.

### Legacy adapter execution

The separate `GeometricIGBundleAdapter` no longer fails on multi-token inputs
while constructing an unused rank-four metric slice. Its private helper now
honestly names and preserves the historical coordinate-smoothing formula:
`s_0 = x_0`, `s_t = x_t - 0.1 * (x_t - x_(t-1))`, using the original predecessor.
Parameter names, shapes and initialization are unchanged.

Six hosted CPU tests cover tiny forward/backward execution, identity initialization,
a nonzero output effect, strict tensor-only module-state roundtrip, and causal
prefixes. They also check that forward does not call the neural connection or
transport methods. Independent one-token smoothing calls do **not** reproduce
a full sequence without previous-coordinate history.

This is **execution evidence, not geometric-law or training certification**.
It does not extend the default-adapter checkpoint loader to this legacy class,
certify historical weights, or validate cached generation, auxiliary losses or
general numerical stability.
[Regression, exact test scope and limits](docs/export-image-audit.md#legacy-adapter-execution-checkpoint).

### Tiny native GGUF execution

Two synthetic F32 Qwen2 files, with 17,136 and 35,296 parameters, now execute in
the pinned native llama.cpp runtime on hosted CPU. Supplied token-ID sequences
exercise grouped-query attention, tied/untied output weights and full/incremental
logits. Against an independent PyTorch/Transformers computation, the largest
observed absolute difference was `1.79e-7`; all argmax IDs matched.
The acceptance tolerances are `rtol=1e-4, atol=1e-5`.

Malformed Q/K tensor shapes are rejected by both the directory inspector and
the native loader. Conversely, negating only output weights leaves the directory
report unchanged except the filename, but changes native logits by `0.77686`.
**A directory fingerprint is not an executed-weight fingerprint.**
[Recorded results and complete synthetic-file hashes](docs/benchmark-results/native-gguf-reference.json)
and [exact scope](docs/export-image-audit.md#tiny-native-gguf-checkpoint).

This is a small F32 runtime control, not execution of the surviving archived
exports, the public converter, text tokenization or quantized kernels.
No adapter is installed; geometric/hypercomplex integration, learned geometry
and task benefit remain unestablished.

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
