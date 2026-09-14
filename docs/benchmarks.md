# Benchmark policy and reproduction protocol

## Current public status

No comparative performance, speed, or learned-geometry figure is currently claimed for IGBundle. Historical numbers in older documents are retained only as unverified historical artifacts.

A fresh adapter-integrity run is available: the unified checkpoint loads exactly against its preserved deployment source and produced finite, non-zero residuals (3.22%–3.57% across four deterministic hidden-state shapes). This verifies numerical execution only; it is not a downstream task result. See [benchmark status](benchmark-status.md).

## A publishable run

Every reported result must contain a committed manifest with the base-model and adapter checkpoint revisions, tokenizer and dataset revisions, command, environment, hardware, seed, decoding configuration, raw predictions, and a paired base-model run under identical settings. Training-dependent claims additionally require at least three seeds with uncertainty.

Kernel tests, telemetry, and task quality must be reported separately. Kernel correctness does not imply model improvement.

## Minimum first benchmark

Before a full evaluation, run base and adapter on 25 examples each from ARC-Challenge, TruthfulQA MC2, and GSM8K with identical settings. Save raw outputs locally, review format and leakage failures, then publish a manifest and an aggregate table linked to stable raw artifacts.

## Geometry claims

Report curvature only after a weight-sensitivity test at fixed coordinates. Report the estimator, trainable parameters, and that test result. Never describe a fixed-kernel check, entropy value, or controller trace as learned geometric transfer.
