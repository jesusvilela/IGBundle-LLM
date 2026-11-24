import sys, traceback
from experiments.validate_lr_warmdown import run_experiment, RunConfig

cfg = RunConfig(
    name="control_lr3e4_wd03",
    base_lr=3e-4,
    fiber_lr=5e-3,
    warmup_ratio=0.1,
    warmdown_ratio=0.3
)
try:
    run_experiment(cfg, "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600", "experiments/lr_warmdown_ab", max_steps=1, eval_every=100)
except Exception as e:
    traceback.print_exc()
