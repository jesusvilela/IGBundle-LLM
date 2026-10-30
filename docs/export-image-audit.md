# GGUF exports, research images, and the geometry boundary

**Inspection: 30 October 2026.** This extends the evidence audit beyond the public README into the local research archive. It establishes properties of the inspected files and generators, not a new model benchmark. The archive's active research checkout was left unchanged.

## What survives in the two inspected GGUFs

| Archive-relative file | Bytes | Architecture / shape metadata | Tensor directory |
|---|---:|---|---|
| `igbundle-llm/dist/base_model.gguf` | 8,098,524,864 | `qwen2`, name `Igbundle_Qwen7B_Cp600`, 28 blocks, width 3584 | 339 tensors: 198 Q8_0 and 141 F32 |
| `igbundle_arc_deployment/qwen_arc_base_0.5B.gguf` | 994,156,160 | `qwen2`, name `Igbundle_Qwen05B_Base`, 24 blocks, width 896 | 290 tensors: 169 F16 and 121 F32 |

Both use GGUF v3. Their names match the expected ordinary Qwen2 tensor-name sets, with an explicit output weight in the first export and none in the second. No additional tensor names or metadata keys matched the published geometry/algebra marker pattern. Tensor offsets, quantization block divisibility, non-overlap, and extents within the file were checked.

The inspector read **5,953,216 + 5,947,510 bytes** of metadata and tensor directories, and **zero tensor-payload bytes**. [cp600-named directory](benchmark-results/gguf-dist-directory.json) · [0.5B directory](benchmark-results/gguf-arc-directory.json). Each report includes every tensor descriptor and a SHA256 of the precise prefix read. **These are not full-model checksums.** Weights with identical headers can differ.

These are not the October cp300 Q4_K_M benchmark artifacts. A metadata name is not immutable checkpoint provenance. Ordinary weights may retain effects of LoRA or other training; the absence of extra names neither disproves such effects nor proves geometry is executed at inference.

### Export/runtime trace

The inspected `export_refined_package.py` converts a cp600 HF directory to Q8_0 and copies `adapter_refined.pt` separately. The packaged `dist/run_inference_integrated.py` fast path calls `llama_cpp.Llama` without loading that adapter. Its alternative PyTorch path is a different loader: it permits missing weights and non-strict loading and hardcodes a base-model identifier. It is not a verified restoration contract.

The ARC package's README describes a 7B export, but its surviving GGUF directory reports width 896 and 24 blocks, consistent with the named 0.5B artifact. A nearby 7B adapter/projector does not resolve that mismatch. No pickle checkpoint was deserialized to guess compatibility.

The repository's `export_gguf.py` merges PEFT/LoRA weights. It is **not** an exporter for arbitrary nonlinear geometric modules. GGUF is a container; runtime architecture support determines which operations execute. A custom geometry-aware runtime is possible, but none is established by these two files and their inspected fast path.

### Public exporter preflight correction

The public exporter previously allocated its base before invoking an absent,
working-directory-relative `safe_convert_wrapper.py`, and reused a fixed merged
directory. It now checks the local PEFT config/weight files, requires an exact
base-identifier match and explicit trusted converter path, and rejects existing
output destinations before model allocation.

`--lora-only` is mandatory acknowledgement that nonlinear IGBundle operators are
omitted. `--dry-run` performs only the stdlib preflight; it creates no outputs and
imports no model backend. Execution reserves fresh output paths, uses PEFT safe
merge, and runs the supplied converter with explicit arguments. Conversion errors
propagate; partial outputs are retained, not silently deleted or reused.
Successful conversion must also pass the existing bounded GGUF directory
inspection and contain tensors. This checks structure and extents, not tensor
values, checkpoint identity, numerical equivalence or runtime behavior.

Nine dependency-free tests use fake model APIs and tiny synthetic GGUF data.
They establish preflight, argument wiring, failure propagation and output guards.
No actual PEFT model was merged and no external converter or model runtime was
validated. The archived packaged runner remains unchanged and uncertified.

## Images: illustration, synthetic data, or measurement?

Nine PNGs were visually inspected and fingerprinted. The [artifact manifest](benchmark-results/artifact-audit.json) contains paths, image dimensions, SHA256 hashes, classifications, and line-numbered excerpts from the inspected source files. Source-to-image associations are supported by filenames and content, not immutable run receipts; the inspected source is not asserted to be the exact source at image creation.

| Image or family | Source finding | Permitted interpretation |
|---|---|---|
| Thesis figure 4: loss / dispersion | `thesis_figures.py:91-121` constructs exponential curves plus random noise | Synthetic illustration, not training telemetry |
| Thesis figure 5: affinities | `thesis_figures.py:123-147` fills blocks with random values | Synthetic block structure, not learned clustering |
| Thesis figure 7: spectrum | `thesis_figures.py:149-174` supplies power laws and a cutoff | Not a measured SVD or evidence against rank collapse |
| `riemannian_geometry.png` | Downward paraboloid, assigned negative heatmap, manually rotated arrows | Curvature sign is wrong for the drawn surface; transport is schematic |
| `information_geometry.png` | Random `A A^T`, hand-written paths, prescribed exponential loss curves | Not a measured Fisher matrix or evidence for “50–75% step reduction” |
| `before_after_comparison.png` | Diagram labels claim “corrected mathematical rigor” | Design illustration, not certification of geometry, natural gradients, or sheaf laws |
| `benchmark_graphical_report.png` | AIME is compared to MATH; ARC-Challenge is labeled ARC-AGI; heuristic fallbacks exist | Unmatched historical visualization, not a valid comparative result |
| `igbundle_topology_riemannian_final.png` | Inspected renderer prescribes `curvature=-1` and permits random embeddings when weights/projections are absent | Layout, not learned hyperbolicity; actual checkpoint binding unresolved |
| LLMOS neural-information-geometry summary | Associated source uses a small MLP with synthetic data; its “curvature” is a log cross-ratio | Separate toy experiment, not evidence about an IGBundle LLM |

The three public synthetic thesis PNGs now contain prominent labels within the image itself. The original archive files remain untouched; their fingerprints are preserved. Other archived images and PDFs are not retroactively made empirical by the word “corrected.”

The separate `generate_visual_report.py` also uses zero-valued mock baselines and `or`-based metric selection, which can replace a legitimate zero. It is not a valid source of matched performance claims.

## Correct geometry, without claiming it was learned

![Analytic curvature reference; no checkpoint data](assets/geometry-reference.svg)

For a graph `z=f(x,y)` with its **Euclidean induced metric**, Gaussian curvature is

```math
K = \frac{f_{xx}f_{yy}-f_{xy}^2}{(1+f_x^2+f_y^2)^2}.
```

In the archived drawing, `f=-(x²+y²)/4`. Thus `f_xx=f_yy=-1/2`, `f_xy=0`, and

```math
K = \frac{1}{4(1+(x^2+y^2)/4)^2} > 0.
```

The surface opens downward but does **not** have negative intrinsic curvature. Replacing it by `f=(x²-y²)/4` changes the numerator to `-1/4`: a negatively curved saddle, but not a constant-curvature hyperbolic plane.

The Poincare disk instead specifies the intrinsic metric

```math
ds^2=\frac{4(dx^2+dy^2)}{(1-x^2-y^2)^2},\qquad x^2+y^2<1.
```

Set `u=log 2-log(1-r²)`. Then `Delta u=4/(1-r²)²` and `K=-exp(-2u) Delta u=-1`. These are analytic identities, with exact rational-arithmetic regression examples; no trained model is involved. The diagram's circular arcs meet the boundary orthogonally because their squared center distance equals one plus their squared radius in unit-disk coordinates.

The LLMOS source's log-distance-cross-ratio has an immediate counterexample as a sectional-curvature estimator: on the flat real line, `(p1,p2,p3,p4)=(0,1,2,3)` gives ratio `4`, while `(0,2,1,3)` gives `1/4`. Its logarithm changes sign under reordering the same flat points. It does not determine curvature.

## Hyperbolic, high-dimensional, and hypercomplex are distinct

| Layer | Evidence required | Current boundary |
|---|---|---|
| Prescribed hyperbolic operator | Defined metric/domain and correct distance/transport laws | The default Poincare distance has bounded regressions; this says nothing about a particular GGUF |
| Learned geometric organization | Checkpoint-bound activations, declared diagnostic, controls and reproducible analysis | Not established by an imposed layout, scalar dispersion, or these images |
| Hypercomplex algebra | A specified product/conjugation, applicable algebraic laws, and an executed model integration path | Not established in the inspected IGBundle source/runtime/export paths |
| Model benefit | Matched task evaluation with provenance, budgets and uncertainty | Remains an open hypothesis |

The manifest records a bounded name scan of 70 Python files in the active `src/igbundle` tree and 48 in the packaged `dist/igbundle_lib/igbundle` copy. It found no listed hypercomplex-algebra identifiers. This is **not** proof that no equivalent implementation exists elsewhere or under other names. High-dimensional real vectors, Hamiltonian dynamics, categorical fibers, and the word “quantum” do not on their own specify quaternionic or octonionic multiplication. Results from separate repositories cannot certify this export without an integration trace.

## Reproduce the bounded checks

### Exact algebra reference

The independent `scripts/algebra_reference.py` defines real, complex, quaternion
and octonion products recursively in dimensions 1, 2, 4 and 8:

```text
(a,b)(c,d) = (ac - conjugate(d)b, da + b*conjugate(c))
conjugate(a,b) = (conjugate(a), -b)
N(x) = sum_i x_i^2
```

Its finite certificate uses Python integers, not floating-point tolerances.
The product is bilinear by construction and conjugation is linear.
Consequently, unit and conjugation-reversal identities reduce to basis checks;
associativity reduces to all basis triples. Alternativity is checked by the
two antisymmetry identities of the trilinear associator. In characteristic zero,
these imply `[x,x,y]=[y,x,x]=0` for every real vector.

Norm composition is not inferred from selected numerical examples: the checker
collects every coefficient of `N(x*y) - N(x)*N(y)` and requires all to vanish.
The quaternion product also has an independent Hamilton-formula regression.
For octonions, basis indices `[1,2,4]` give left product `e7`, right product `-e7`,
and associator `2*e7`. Reassociation is therefore invalid in general.

The [reproducible certificate](benchmark-results/algebra-reference.json)
covers only this explicitly defined algebra, not learned weights, an intrinsic
hyperbolic metric, neural integration or a GGUF runtime. It is not a new
mathematical discovery or a formal proof-assistant certificate.
Private cross-project materials were not copied into the public evidence set.

### Inspection commands

From the repository, with Python 3.10+ and no third-party packages:

```powershell
python scripts\inspect_gguf.py H:\LLM-MANIFOLD\igbundle-llm\dist\base_model.gguf --output cp600-directory.json
python scripts\inspect_gguf.py H:\LLM-MANIFOLD\igbundle_arc_deployment\qwen_arc_base_0.5B.gguf --output arc-directory.json
python scripts\export_artifact_audit.py --archive H:\LLM-MANIFOLD --output artifact-audit.json
python -m unittest discover -s tests -p test_gguf_inspection.py
python -m unittest discover -s tests -p test_artifact_audit.py
```

Output paths for the inspection/export commands must be new. The source/image exporter refuses to reuse reviewed classifications when those source or image fingerprints have changed; re-audit them first. The GGUF inspector refuses unsupported formats/types, malformed/truncated directories, overlaps, bad alignment, and a metadata/directory read above 32 MiB. It does not memory-map weights, import a model backend, or run inference. Tests use tiny synthetic GGUF fixtures; CI does not contain the original models and cannot re-inspect their payloads.

`scripts/geometry_reference.py` reproducibly renders the analytic SVG. On Windows, `scripts/label_archived_figures.ps1 -Archive H:\LLM-MANIFOLD -OutputDirectory <existing-output-directory>` adds labels to the three original thesis PNGs after verifying their archived hashes; it refuses to overwrite an archive source.

## Default-adapter restoration checkpoint

The public HF entry points had another concrete failure mode: missing or partial
weights could leave freshly initialized adapters running while appearing to load
a checkpoint. `src/igbundle/integrations/checkpoint.py` now rejects incomplete
default-adapter state.

- `app.py`, `benchmark.py`, `bench_arena.py`, and `validate_effects.py` require
  `adapter_weights.pt`; `evaluate.py` requires `full_state_dict.pt`.
- Missing files are rejected before base-model allocation in those entry points
  (including the `validate_effects.py` CLI preflight).
- Every registered `IGBundleAdapter` tensor must be present at its exact name,
  with the expected shape, finite values, and a compatible dtype. Conversion
  overflow is rejected. All adapter tensors are validated before any are copied.
- Adapter-only mappings reject extra keys. Explicit full-state mode permits
  additional known model keys but **does not restore them**.
- Disk reads use `torch.load(..., map_location="cpu", weights_only=True)`, with no
  fallback to unrestricted pickle loading.
- Declared PEFT load failures are no longer swallowed. The app's CPU branch now
  attempts its declared PEFT component rather than silently omitting it.

The bounded tests restore tiny synthetic tensor dictionaries, verify that base
weights remain unchanged, and reject missing, extra, malformed, nonfinite,
meta, sparse, and incompatible tensors. A controlled output bias verifies that
the restored adapter executes inside a block wrapper and receives gradients.
Entry-point tests execute isolated loader functions with stubs; they do not
import Unsloth, Transformers or PEFT, or load a real LLM.

This contract intentionally rejects legacy partial or differently prefixed
checkpoints rather than guessing their meaning. It covers the **default**
adapter only, not the separate `GeometricIGBundleAdapter` or packaged archive.
Tensor coverage cannot identify a base checkpoint, recover training settings,
certify PEFT loading, or prove that the weights were learned. No original adapter
pickle was opened, no historical run was reconstructed, and no new task score
was obtained. The separate demo correction below addresses telemetry labels;
legacy benchmark protocols remain outside this restoration contract.

Run `python -m unittest discover -b -s tests -p test_checkpoint.py` in an existing
PyTorch environment. The hosted evidence workflow uses CPU-only PyTorch and one
thread. A local PyTorch import exceeded a two-minute wait and was stopped;
tensor validation for this checkpoint ran on GitHub, not as a local model job.

## Demo evidence checkpoint

The public `app.py` at `17fe6dfddad8fb2a99ca9a89e00c3fdf55f7cacf`
still displayed the withdrawn +131.5% and 94.2% claims. Its missing-log branch
constructed exponential training curves; ablation accuracies were literals.
`load_topo_stats()` also supplied invented fallback statistics or displayed
unbound `thesis_stats.json` contents. These were display paths, not new experiments.

The corrected app removes those substitutions and unsupported narrative claims.
Training and ablation panels now render **unavailable** placeholders with no
data curves or bars. Local JSON files are not automatically promoted to evidence.
The standalone `generate_braintop_viz.py` renderer was outside the demo checkpoint;
the app no longer imports it or directs users to its output as learned geometry.
Its subsequent replacement is documented below.

`MixtureState.sigma` is a Gaussian standard deviation, and `MixtureState.p` is a
categorical probability tensor. The corrected live labels reflect those definitions:
no curvature estimate, pairwise affinity, or geometric-quality verdict is inferred.
The open-interval sigma count `(0.5, 2.0)` is retained only as an explicitly
arbitrary descriptive counter, not a target or “MFR” result.

`src/igbundle/demo_evidence.py` computes observation-weighted summaries across
layers and forward calls. It supports varying sequence lengths; repeated context
is not deduplicated. Probability sums within 0.005 of one are normalized to account
for low-precision softmax rounding; larger deviations, nonfinite values, invalid
shapes, and nonpositive sigma are rejected. Zero categorical probabilities use
the `0 log 0 = 0` convention.

Only the first 4,096 sigma values are retained for the explicitly labeled prefix
histogram. All observations contribute to aggregate counters; no unbounded
generation history or activation arrays are kept. A generation lock separates
requests, and capture is reset before each request. Missing states produce an
unavailable message, not `NaN`, zero accuracy, or a quality badge.

The headless tests use tiny arrays and isolated app functions. They render the
plots, assert that unavailable panels contain no curves/bars, check variable-length
aggregation, bounded storage, reset behavior and hook cleanup. A separate check
constructs the real Gradio interface with a stubbed model and intercepted launch.
The tests do not import Transformers, PEFT or Unsloth, start a server, load models,
or exercise the real inference backend. Optional UI dependencies are listed in
`requirements-demo.txt`; the CLI now requires an explicit checkpoint.

## Projection-display checkpoint

At `a1ac1a4fc2888413951c169e81d469367a83823c`, the standalone renderer
substituted random projection rows when weights were unavailable, randomly
subsampled lite displays, and selected a purported manifold from checkpoint or
output filenames. It relied on version-dependent `torch.load` defaults and
silently selected the first matching projection. None of those operations
measured learned topology.

The retained entry point now writes a self-contained HTML/SVG parameter heatmap.
It requires `adapter_weights.pt`, reads at most 64 MiB plus one overflow-detection
byte, and uses CPU `weights_only=True` loading without an unsafe retry.
Only an exact `input_proj.weight` suffix is eligible; multiple candidates require
an explicit `--tensor-key`. The selected tensor must be a nonempty, finite,
dense floating matrix with at most 4,194,304 values.

The display is the first 64 rows and columns, or 32 in `--lite` mode.
It is a deterministic prefix, not a representative sample, activation map or
dimensionality reduction. Rows are output channels, columns input coordinates.
Colors use the displayed prefix's maximum absolute value; scales differ between
images. The exact input-file SHA-256, selected key, full shape and dtype are
embedded in the HTML. The SVG itself carries a parameter-only warning.

No base model, topology builder, random layout or semantic labels are used.
The historical `node_metadata` argument now fails explicitly instead of promoting
unbound annotations to evidence. Missing/malformed inputs produce errors, not
scaffolds or success-shaped output. Existing output survives validation failures.

Tiny synthetic checkpoint tests cover exact selection, rejected loads without
retry, finite/dense dtype checks, byte/value/display limits, deterministic output,
zero/BF16 display and HTML escaping. They do not open historical checkpoints.
The file-size guard bounds the input read, not arbitrary deserialization resource
use; this utility is for trusted local files. Selecting one tensor does not
validate the remaining state, restore the complete adapter or certify checkpoint
provenance. No new hyperbolicity, hypercomplexity or task result is claimed.

## Coverage and remaining blockers

### Tiny real Transformers/PEFT integration

Five hosted CPU checks use PyTorch 2.6.0, Transformers 4.48.3, PEFT 0.14.0 and
Accelerate 1.2.1. The randomly initialized Qwen2 has two layers, hidden width 16,
vocabulary 32 and fewer than 100,000 base parameters. No pretrained model or
dataset is downloaded.

The checks cover exact identity logits at initialization; real PEFT save/reload
before wrapper insertion; strict serialized adapter restoration with an observable
logit effect and unchanged base/LoRA tensors; finite gradients reaching both
adapter and LoRA; and two-token greedy generation. Cached final-token logits
match full-prefix logits at `rtol=1e-5, atol=1e-6`, with nonzero adapter values.

Two coupled wrapper defects were repaired: registered modules are retrieved through
`nn.Module` lookup, and CPU adapters remain float32 even on a host with a
BF16-capable GPU. The latter is checked with a mocked capability probe.

This validates a narrow, real backend composition, not the five public loader
entry points end-to-end. The subsequent LoRA contract below adds wrapped-save
coverage. No historical weights, quantization, pretrained scores, training
convergence, full server workflow or geometry-aware GGUF inference is certified.

### Exact declared-LoRA contract

The six public load/merge paths now use a shared exact-state check: app,
evaluation, benchmark, arena, effect validation and export. The optional Unsloth
app path is checked after its loader returns; that backend itself was not run.
Only one default LoRA adapter is supported.

The checker reads local safetensors on CPU, or a tensor-only
`adapter_model.bin` with `weights_only=True`. It compares PEFT's serialized-key
view with the saved mapping: no missing/extra keys, malformed tensors,
nonfinite values, conversion overflow or unequal restored values are accepted.
It does not remap `original_block` prefixes.

Tiny real PEFT checks demonstrate the distinction: a save made after wrapper
insertion is rejected against an unwrapped base, and its nonzero LoRA state
restores when the wrapper structure matches. Tests also reject partial/extra
state, wrong shapes, sparse/nonfinite values and overflow.

This is a post-load assertion, not an atomic rollback. On failure, callers must
discard the partially constructed model. Paths that load PEFT before wrapping
and paths that wrap first require different matching checkpoint names.
This deliberate failure is safer than silently running initialized LoRA values;
it does not repair or certify any historical checkpoint.

### Hyperbolic transport checkpoint

The expanded read-only audit traced
`igbundle-llm/src/igbundle/geometry/poincare.py` into
`src/igbundle/dynamics/hamiltonian.py` and checked the existing
`tests/test_audit_fixes.py`. This is the local archive, not the public default
adapter. A newer, separately modified local conformal-connection implementation
was also encountered; it was neither overwritten nor certified.

| Inspected archive file (relative to `igbundle-llm`) | SHA256 |
|---|---|
| `src/igbundle/geometry/poincare.py` | `2bb9e137911eb84e8a628df7b46baef5b7009103af2f6022d6b750cbd2d28638` |
| `src/igbundle/dynamics/hamiltonian.py` | `64d049e0b95cf3fbd3f5f3208a5ee65b86641ad92d612448375777e60f966f4a` |
| `tests/test_audit_fixes.py` | `a1bdf432d4ab423393f5f29e4956c4b5f5a238e5b51a9c23ea0a3573d6397ad5` |

The archived Poincare method computes only
`v_y = (lambda_x / lambda_y) v_x`. This preserves metric inner products for
`g_x = lambda_x^2 I`, but every closed-loop product telescopes to the identity.
It therefore cannot reproduce nonzero Levi-Civita holonomy. Length preservation
alone would miss this defect. The inspected audit tests cover the corrected
curvature-scaled distance/logarithm domain, not this missing rotation.

The new opt-in transport reference uses, for fixed `c > 0`,

```text
lambda_x = 2 / (1 - c ||x||^2),   g_x = lambda_x^2 I,   K = -c
PT_x->y(v) = (lambda_x / lambda_y) gyr[y, -x](v)
CoPT_x->y(p) = (lambda_y / lambda_x) gyr[y, -x](p)
```

Here `gyr` is the orthogonal gyrovector rotation, evaluated after scaling
coordinates by `sqrt(c)`. The second map follows from inverse-transpose duality:
if the tangent map is `s R` with orthogonal `R`, its covector map is `R / s`.
It preserves the pairing `p(v)` and the covector kinetic energy
`p^T g^{-1} p / 2` under pure transport.

The archived integrator uses precisely that inverse-metric kinetic energy but
passes its momentum directly to tangent transport. For the isolated test edge
`(.3,0) -> (0,.4)`, that substitution gives a kinetic-energy ratio of
`0.7260249991`, whereas the dual map preserves it within float64 roundoff.
This is an operator counterexample, **not an energy measurement or repair of
the complete integrator**. No symplecticity or time-reversal claim is made.

For the positively oriented geodesic triangle
`(0,0) -> (.3,0) -> (0,.4) -> (0,0)`, the full returned tangent is
`(0.9716088328, -0.2365930599)` and its signed rotation is
`-0.2388578520` radians. At `K=-1`, this equals the negative angle defect
`-2 atan(.3*.4)`; the tests also compute the three interior angles independently
on the hyperboloid. The scale-only formula returns `(1,0)`.

An independent check embeds the ball into the Lorentz hyperboloid
`<X,X>_L = -1/c`, pushes a tangent through the chart Jacobian, applies
`V -> V + <Y,V>_L (X+Y)/(1/c - <X,Y>_L)`, and differentiates the inverse chart.
The fixed loop agrees within `3.34e-16` absolute error. Its angle error is below
`8.33e-17`. The predeclared witness acceptance tolerance is `1e-12`.

[Published numerical witness](benchmark-results/transport-reference.json).
The first execution used hosted CPU PyTorch 2.6.0 in
[run 41139560935](https://github.com/jesusvilela/IGBundle-LLM/actions/runs/41139560935).
Bounded tests use dimensions 2, 3 and 8; `c=.25,1,4`; 12 seeded point pairs
per configuration at scaled radii .6 and .4; float64 reference comparisons
with `rtol=atol=1e-12`; inverse and metric-product checks; dual pairing and
kinetic energy; finite-difference gradient checks including the origin;
batch broadcasting and half/bfloat16 promotion to float32; and explicit
invalid-input rejection. This is not exhaustive floating-point verification,
especially near the ball boundary, or a formal proof-assistant certificate.

```powershell
python -m unittest discover -b -s tests -p test_poincare_transport.py
python scripts\transport_reference.py
```

Inputs must be finite, broadcast-compatible tensors with matching dtype/device,
and points strictly inside the curvature-scaled ball. The implementation does
not silently project or clamp points and raises on transport overflow.
The class intentionally provides only conformal factor and tangent/cotangent
transport, not the archive's full exp/log/dynamics API. It is not imported by
either adapter or GGUF runtime. Archive files remain unchanged. The metric is
prescribed, not learned; this adds no hypercomplex multiplication or task result.

Inventory visited 1,577 workspace directories while excluding dependencies, Git internals, reparse-point directories and duplicate audit checkouts. Vendored llama.cpp model folders were checked separately and contained tokenizer fixtures. Dataset pictures and UI/vendor graphics were not treated as scientific evidence. Only the nine selected PNGs above received visual review; this is not a certification of every image or document in the archive.

No historical model payload, historical adapter pickle, full multi-gigabyte hash, pretrained-model inference, training job, or GPU workload was used. Tiny synthetic checkpoints and random-model CPU execution were checked on the hosted runner. No private cross-repository content was republished. The blockers to stronger claims remain exact checkpoint/runtime bindings, restoration of the actual historical companion, tested hypercomplex integration, and a resource-approved matched model experiment.
