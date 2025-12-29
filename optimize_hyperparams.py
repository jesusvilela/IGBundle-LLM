import optuna
import os
import sys
import torch
import gc
from transformers import TrainingArguments, Trainer
from train import load_model_tokenizer # Re-use loading logic if possible, or redefine
# Note: We need to import carefully to avoid running the main block of train.py if it has one.
# Since train.py has `if __name__ == "__main__": train()`, we are safe to import functions if we refactored.
# However, train.py is a script. I'll redefine the minimal loading logic here to be safe and standalone.

import argparse
import yaml
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from igbundle.modules.adapter import IGBundleAdapter
from igbundle.integrations.hf_patch import wrap_hf_candidate

def objective(trial):
    # 1. Hardware Check (Optional safety)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for optimization study.")
        
    # 2. Sample Hyperparameters
    # We focus on the Sheaf/IGBundle parameters
    alpha_trial = trial.suggest_float("alpha", 0.5, 2.0) # Base KL
    beta_trial = trial.suggest_float("beta", 0.5, 2.0) # Fiber KL
    bottleneck_trial = trial.suggest_categorical("bottleneck_dim", [128, 256, 512])
    lr_trial = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    
    print(f"\n[Optuna] Starting Trial {trial.number}")
    print(f"Params: alpha={alpha_trial}, beta={beta_trial}, bn={bottleneck_trial}, lr={lr_trial}")
    
    # 3. Setup Config (Modify base config)
    with open("configs/qwen25_7b_igbundle_lora.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['ig_adapter']['alpha'] = alpha_trial
    cfg['ig_adapter']['beta'] = beta_trial
    cfg['ig_adapter']['bottleneck_dim'] = bottleneck_trial
    cfg['training']['learning_rate'] = lr_trial
    cfg['training']['output_dir'] = f"output/optuna/trial_{trial.number}"
    cfg['training']['max_steps'] = 20 # Short run for proxy metric
    cfg['training']['save_steps'] = 100 # Don't save often
    
    # 4. Run Training (Subprocess is safer to clean VRAM entirely between trials)
    # We will write a temp config and call train.py
    temp_config_path = f"configs/optuna_temp_{trial.number}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(cfg, f)
        
    import subprocess
    cmd = [
        sys.executable, "train.py", 
        "--config", temp_config_path,
        "--smoke_test" # Add a flag to train.py to just return loss? Or check logs?
    ]
    
    # NOTE: Since we cannot easily modify train.py to return values to this process without
    # pipes or file parsing, we will parse the output log or a metrics file.
    # For now, we assume train.py writes to trainer_state.json
    
    try:
        # We assume the user runs this passing "task.md" status checks if desired, 
        # but for now we just try to run it.
        # WARNING: Running this while another training is active WILL fail.
        # We catch the error and prune the trial.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Trial failed: {result.stderr}")
            raise optuna.TrialPruned()
            
        # Parse result from the output folder
        state_path = os.path.join(cfg['training']['output_dir'], "trainer_state.json")
        import json
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                data = json.load(f)
                # Get last loss
                if data['log_history']:
                    final_loss = data['log_history'][-1].get('loss', 100.0)
                    return final_loss
        
        return 100.0 # Default bad loss
        
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    print("Initializing Optuna Study...")
    print("WARNING: This script launches training subprocesses.")
    print("Ensure no other heavy GPU tasks are running.")
    
    study = optuna.create_study(
        study_name="igbundle_optimization",
        direction="minimize",
        storage="sqlite:///igbundle_optuna.db",
        load_if_exists=True
    )
    
    try:
        study.optimize(objective, n_trials=10)
        print("Best params:", study.best_params)
    except Exception as e:
        print(f"Optimization interrupted: {e}")
