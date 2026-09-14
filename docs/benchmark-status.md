# Phase 8 benchmark status

## Candidate

- Base model: `H:\LLM-MANIFOLD\igbundle_qwen7b_cp600`
- Geometric checkpoint: `H:\LLM-MANIFOLD\igbundle_phase8_training\checkpoint-3000\adapter_weights.pt`
- Checkpoint SHA-256: `2004373636B049FC03771EE087FF1E4053D8003C18C8F3FA94668D598145DB14`
- Adapter configuration: Qwen hidden size 3584; 8 components; latent dimension 64; 16 categories; dynamics and geodesic attention enabled; `vision_dim=1152`.

The checkpoint loads into `GeometricIGBundleAdapter` with zero missing and zero unexpected keys under this configuration. A text-only configuration drops 17 vision parameters; the default 768-dimension vision configuration is incompatible. Neither is valid for the paired run.

## Current execution result

No task-quality figure has been produced. Two 4-bit GPU loads terminated at shard 96 of 339 without a Python traceback. A 5 GiB GPU cap fails earlier because the installed bitsandbytes path rejects CPU-dispatched 4-bit modules. At the time of execution the host exposed about 6.7 GiB free GPU memory and 7.78 GiB free RAM; the four base-model shards total about 15.1 GB.

## Next execution gate

Run `scripts/benchmark_phase8_smoke.py` on a host that can load the paired base and adapter. It records raw greedy-decoding outputs for identical base and geometric-adapter prompts. Inspect that artifact before launching scored ARC-Challenge, TruthfulQA MC2, and GSM8K evaluation. This checkpoint's legacy curvature telemetry must remain separated from downstream task scores because its historical fixed-conformal estimator was shown weight-invariant.
