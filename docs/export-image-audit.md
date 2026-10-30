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
was obtained. The app's historical telemetry labels and legacy benchmark
protocols remain outside this verified contract.

Run `python -m unittest discover -b -s tests -p test_checkpoint.py` in an existing
PyTorch environment. The hosted evidence workflow uses CPU-only PyTorch and one
thread. A local PyTorch import exceeded a two-minute wait and was stopped;
tensor validation for this checkpoint ran on GitHub, not as a local model job.

## Coverage and remaining blockers

Inventory visited 1,577 workspace directories while excluding dependencies, Git internals, reparse-point directories and duplicate audit checkouts. Vendored llama.cpp model folders were checked separately and contained tokenizer fixtures. Dataset pictures and UI/vendor graphics were not treated as scientific evidence. Only the nine selected PNGs above received visual review; this is not a certification of every image or document in the archive.

No historical model payload, adapter pickle, full multi-gigabyte hash, model inference, training job, or GPU workload was used. Only tiny synthetic tensor checkpoints were deserialized for restoration tests. No private cross-repository content was republished. The blockers to stronger claims are concrete: exact checkpoint/runtime bindings, restoration of the actual historical companion, implemented and tested hypercomplex operations in that path, and a resource-approved matched model experiment. The new default-adapter contract does not manufacture those missing results.
