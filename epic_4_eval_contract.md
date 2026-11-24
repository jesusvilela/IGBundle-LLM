# EPIC 4: Evaluation and Benchmark Contract

## Objective
Define the exact post-checkpoint evaluation battery for EOS, entropy, prompt leakage, degeneration loops, and benchmark prompts. This contract serves as the strict gate for checkpoint promotion.

## Evaluation Battery Matrix

### 1. Natural EOS Ratio
- **Metric:** Percentage of 50 evaluation generations that terminate naturally with a valid EOS token (`<|im_end|>` or `<|endoftext|>`) before hitting `max_new_tokens`.
- **Pass Threshold:** > 0.64 (to beat current baseline).
- **Target:** >= 0.80.

### 2. Entropy (S) Health
- **Metric:** Mean S calculated across the 50 evaluation generations (or measured over the evaluation dataset).
- **Pass Threshold:** S_mean >= 0.60.
- **Target:** S_mean in [0.80, 1.40].

### 3. Prompt Leakage
- **Metric:** Percentage of eval generations containing known prompt leakage markers (e.g., `_LEAKAGE_FRAGMENTS`, `<|im_start|>system`).
- **Pass Threshold:** 0% (Zero tolerance). Any leakage immediately invalidates the checkpoint.

### 4. Degeneration Loops
- **Metric:** Repetition loop detection. Generative sequences ending in repeating n-grams (e.g., repeating the same sentence or phrase > 3 times).
- **Pass Threshold:** < 2% of evaluation samples exhibiting degeneration loops.

### 5. Benchmark Prompts (Quality)
- **Metric:** Keyword scoring and constraint satisfaction on the 8 calibration prompts used in `run_gsp_benchmark.py`.
- **Pass Threshold:** Average answer quality score >= 80% of the baseline model's score.

## Reporting Requirements
Each checkpoint evaluation must output a structured `eval_metrics.json` containing:
- `step`: Checkpoint step.
- `eos_ratio`: Float [0, 1].
- `entropy_mean`: Float.
- `leakage_rate`: Float [0, 1].
- `degeneration_rate`: Float [0, 1].
- `benchmark_score`: Float.
- `promotion_status`: Boolean (True if all Pass Thresholds are met).