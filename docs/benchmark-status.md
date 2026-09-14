# Benchmark status

## Candidate

- Base model: `H:\LLM-MANIFOLD\igbundle_qwen7b_cp600`
- Geometric checkpoint: `H:\LLM-MANIFOLD\igbundle_phase8_training\checkpoint-3000\adapter_weights.pt`
- Checkpoint SHA-256: `2004373636B049FC03771EE087FF1E4053D8003C18C8F3FA94668D598145DB14`
- Adapter configuration: Qwen hidden size 3584; 8 components; latent dimension 64; 16 categories; dynamics and geodesic attention enabled; `vision_dim=1152`.

The checkpoint loads into `GeometricIGBundleAdapter` with zero missing and zero unexpected keys under this configuration. A text-only configuration drops 17 vision parameters; the default 768-dimension vision configuration is incompatible. Neither is valid for the paired run.

## Current execution result

No task-quality figure has been produced. The local 7B base model terminated during 4-bit loading without a Python traceback (at shard 96 of 339 in one Transformers attempt and after the first shard in both direct-GPU Transformers and Unsloth 2026.4.5 attempts). A 5 GiB GPU cap fails earlier because the installed bitsandbytes path rejects CPU-dispatched 4-bit modules. At the time of execution the host exposed about 6.7 GiB free GPU memory and about 8 GiB free RAM; the four base-model shards total about 15.1 GB. There is no locally available smaller base with the required 3584-wide hidden state.

## Completed adapter-integrity run

The executable checkpoint is `igbundle_unified_training/final/adapter_weights.pt` (SHA-256 `ccea55d142cc8d962966011213bb9db5f90e2f9436ba847f6dba1c5edff0328e`) paired with the preserved `igbundle_unified_deployment/src` source snapshot. It loaded strictly with **0 missing and 0 unexpected tensors**.

On an RTX 3060 Ti, CUDA 11.8, PyTorch 2.7.1, the fresh deterministic numerical-integrity run produced finite outputs on all four hidden-state shapes. The residual norm ratio `||Δh|| / ||h||` was **3.22%–3.57%** (four seeds; shapes from `1×1×3584` through `2×32×3584`). Base coordinates had shape `B×T×8×64`; fiber sections had shape `B×T×8×16`. This establishes numerical execution and a non-inert effect only; it is not language-task accuracy, speed, or a geometry-quality result. Raw artifact: `H:\LLM-MANIFOLD\benchmark-runs\unified-integrity-2026-09-14\results.json`.

The historical deployment implementation requires autograd inside its Hamiltonian step even during evaluation. Benchmark runners must not use `torch.inference_mode()` around adapter calls; use `torch.no_grad()` for the base model and a local `torch.enable_grad()` scope for the adapter.

## Checkpoint-selection audit

Controlled GPU execution (seeded random hidden states, shape `1×4×3584`) separates checkpoint executability from model quality:

| Checkpoint | Load result | Residual effect | Benchmark decision |
| --- | --- | --- | --- |
| Phase 8 cp3000 / cp3900 | Exact load: 0 missing, 0 unexpected | Exactly zero: `output_proj` weight and bias are all zero | Do not benchmark: it cannot change generation. |
| Phase 9 Odyssey cp4001 | Text core loads; 6 vision keys missing / 12 spectral-normalization keys unexpected | Finite but `||Δh|| / ||h|| = 2.00` | Do not benchmark until the vision-key conversion and residual calibration are explicit and tested. |
| Unified final | 40 missing / 12 unexpected parameters against current adapter code; **0 / 0** against preserved unified-deployment source | Finite; fresh four-shape run: `||Δh|| / ||h|| = 0.0322–0.0357` | Selected for paired generation only with the preserved exact source; task scoring remains blocked by base-model loading. |

This is a direct falsification result: a checkpoint can deserialize and emit telemetry while being inert, incompatible, or unsafe for downstream generation. A successful benchmark must select a checkpoint only after exact-load and residual-effect gates pass.

## Development priorities

1. Version adapter configuration and code with every checkpoint, then reject non-exact loads by default.
2. Add a regression test that asserts a finite, bounded, nonzero residual for a saved checkpoint; report the residual ratio alongside every benchmark.
3. Preserve spectral-normalization parameters or reconstruct them deterministically during checkpoint migration.
4. Establish and test an inference residual budget (for example, `0 < ||Δh|| / ||h|| ≤ 0.1`) before task evaluation.
5. After the executable checkpoint and memory environment are resolved, run the paired base/adapted smoke suite before full scored evaluation.

## Next execution gate

Run `scripts/benchmark_unified_smoke.py` on a host that can load the paired base and adapter. It records raw greedy-decoding outputs for identical base and geometric-adapter prompts, and rejects non-finite, inert, and overly disruptive adapters. Inspect that artifact before launching scored ARC-Challenge, TruthfulQA MC2, and GSM8K evaluation. Legacy curvature telemetry must remain separated from downstream task scores because its historical fixed-conformal estimator was shown weight-invariant.
