# EPIC 2: Entropy Collapse Guardrails

## Objective
Define explicit anti-collapse controls for S (Entropy), including thresholds, early-stop criteria, and checkpoint acceptance rules to prevent training continuations that improve EOS at the expense of S-health.

## Guardrail Specifications

### 1. Numeric Entropy (S) Thresholds
- **Target Health Band:** [0.80, 1.40]
- **Marginal Band:** [0.60, 0.80]
- **Collapse Threshold:** S < 0.30

### 2. Training Abort Criteria (Early Stop)
- **Moving Average:** Track the rolling average of S over the last 50 training steps.
- **Trigger Condition:** If the 50-step rolling average of S drops below the Collapse Threshold (0.30), the training run must **ABORT** immediately.
- **Action:** Halt training, save an `emergency-collapse` checkpoint for forensic analysis, and return an error state to the orchestrator.

### 3. Checkpoint Acceptance Rules
- During the evaluation loop (e.g., every 200 steps):
  - S must be calculated over the 50 evaluation generations.
  - **Rejection:** If the average eval S < 0.60, the checkpoint is marked as **FAILED** and cannot be promoted, regardless of its Natural EOS ratio.
  - **Promotion:** To beat the baseline (`checkpoint-alignment-eos-1400`), a checkpoint must achieve a Natural EOS ratio > 0.64 AND maintain an average eval S >= 0.60.

### 4. Trainer Behavior Edits Required
- Modify `train_odyssey_v3.py` to maintain a deque of the last 50 `S_actual` values.
- Insert the Early Stop trigger in the training loop after telemetry logging.
- Update the `evaluate()` method to return both the `eos_ratio` and `eval_S_mean`.