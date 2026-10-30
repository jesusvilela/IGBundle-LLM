try:
    from igbundle.utils import triton_fix
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from igbundle.utils import triton_fix

import argparse
import yaml
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from igbundle.integrations.checkpoint import load_weights, require_checkpoint_file, load_peft_checked
from igbundle.modules.losses import SheafLoss

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen25_7b_igbundle_lora.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    full_sd_path = require_checkpoint_file(os.path.join(args.checkpoint, "full_state_dict.pt"))
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model_id'], trust_remote_code=True)
    
    # Load Base
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg['base_model_id'],
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        # bnb_4bit_quant_type="nf4"
    )
    
    # Inject Adapters (Architecture)
    print("Injecting Adapters...")
    # Helper class to mimic config object access
    class DictConfig:
        def __init__(self, d):
            for k,v in d.items(): setattr(self, k, v)
            
    adapter_cfg = DictConfig(cfg['ig_adapter'])
    model = wrap_hf_candidate(model, adapter_cfg)
    
    # Load Weights
    # 1. LoRA
    print("Loading LoRA...")
    model = load_peft_checked(model, args.checkpoint)
        
    # 2. IGBundle Weights
    # If they were saved separately or in the checkpoint
    print(f"Restoring complete IGBundle state from {full_sd_path}")
    load_weights(model, full_sd_path, full_state=True)
    
    model.eval()
    
    # Evaluate Loop
    text = "The nature of consciousness is"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
        
    print("Generated:", tokenizer.decode(outputs[0]))
    
    # Probe Abstraction (Log Sigmas)
    collector = StateCollector()
    collector.attach(model)
    
    with torch.no_grad():
        model(**inputs)
        
    states = collector.states
    print(f"Captured {len(states)} states.")
    for i, s in enumerate(states):
        # s is MixtureState
        mean_sigma = s.sigma.mean().item()
        print(f"Layer {i}: Mean Sigma = {mean_sigma:.4f}")

if __name__ == "__main__":
    evaluate()
