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
from peft import PeftModel
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from igbundle.modules.losses import SheafLoss

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen25_7b_igbundle_lora.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
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
    try:
        model = PeftModel.from_pretrained(model, args.checkpoint)
    except Exception as e:
        print(f"PEFT load failed (maybe full model?): {e}")
        
    # 2. IGBundle Weights
    # If they were saved separately or in the checkpoint
    print("Loading IGBundle weights from checkpoint if available...")
    
    full_sd_path = os.path.join(args.checkpoint, "full_state_dict.pt")
    if os.path.exists(full_sd_path):
        print(f"Loading full state dict from {full_sd_path}")
        sd = torch.load(full_sd_path)
        # Load only adapter keys if base is frozen/4bit
        adapter_sd = {k: v for k, v in sd.items() if "igbundle" in k or "adapter" in k}
        model.load_state_dict(adapter_sd, strict=False)
    
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
