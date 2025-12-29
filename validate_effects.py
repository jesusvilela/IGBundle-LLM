try:
    from igbundle.utils import triton_fix
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from igbundle.utils import triton_fix

import os
import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from safetensors.torch import load_file

def load_base_model(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Loading Base Model: {cfg['base_model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model_id'], trust_remote_code=True)
    
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg['base_model_id'],
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['base_model_id'],
            device_map="cpu",
            trust_remote_code=True
        )
    
    model.eval()
    return model, tokenizer, cfg

def inject_adapter(model, cfg, checkpoint_path):
    print("Injecting IGBundle Adapter...")
    if hasattr(model.config, "hidden_size"):
        cfg['ig_adapter']['hidden_size'] = model.config.hidden_size
        
    class DictConfig:
        def __init__(self, d):
            for k,v in d.items(): setattr(self, k, v)
    adapter_cfg = DictConfig(cfg['ig_adapter'])
    
    # Wrap model
    model = wrap_hf_candidate(model, adapter_cfg)
    
    print(f"Loading Weights from {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        try:
            # 1. Peft/LoRA
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            model = PeftModel(model, peft_config)
            
            # 2. Strict=False load for LoRA + IGBundle
            if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
                w = load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
                model.load_state_dict(w, strict=False)
            elif os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
                w = torch.load(os.path.join(checkpoint_path, "adapter_model.bin"), map_location="cpu")
                model.load_state_dict(w, strict=False)
                
            # 3. Explicit IGBundle weights
            ig_path = os.path.join(checkpoint_path, "adapter_weights.pt")
            if os.path.exists(ig_path):
                print(f"Loading IGBundle specific weights from {ig_path}")
                ig_w = torch.load(ig_path, map_location="cpu")
                model.load_state_dict(ig_w, strict=False)
                
            print("Adapter Loaded Successfully.")
        except Exception as e:
            print(f"Error loading adapter: {e}")
            
    return model

def generate(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_7b_igbundle_lora.yaml")
    parser.add_argument("--checkpoint", default="output/igbundle_qwen7b/checkpoint-60")
    args = parser.parse_args()
    
    # 1. Load Base
    model, tokenizer, cfg = load_base_model(args.config)
    
    prompts = [
        "What is the mathematical definition of a manifold?",
        "Explain the concept of entropy in simple terms.",
        "Write a short poem about coding."
    ]
    
    print("\n=== 1. BASE MODEL GENERATION ===")
    base_results = {}
    for p in prompts:
        print(f"Prompt: {p}")
        out = generate(model, tokenizer, p)
        print(f"Base Output: {out}\n")
        base_results[p] = out
        
    # 2. Inject Adapter
    print("\n=== INJECTING IGBUNDLE ADAPTER ===")
    model = inject_adapter(model, cfg, args.checkpoint)
    
    # Attach collector for Sigma metrics
    collector = StateCollector()
    collector.attach(model)
    
    print("\n=== 2. IGBUNDLE MODEL GENERATION ===")
    ig_results = {}
    metrics = {}
    
    for p in prompts:
        collector.clear()
        print(f"Prompt: {p}")
        out = generate(model, tokenizer, p)
        print(f"IGBundle Output: {out}\n")
        ig_results[p] = out
        
        # Check metrics from last run
        if collector.states:
            # Average sigma across layers and components
            sigmas = [s.sigma.mean().item() for s in collector.states]
            avg_sigma = sum(sigmas)/len(sigmas)
            metrics[p] = avg_sigma
            print(f"  -> Average Internal Sigma (Curvature): {avg_sigma:.4f}")
        else:
            print("  -> No states captured (Adapter inactive?)")
            
    print("\n=== SUMMARY COMPARISON ===")
    for p in prompts:
        print(f"\nPrompt: {p}")
        print(f"BASE: {base_results[p][:100]}...")
        print(f"IGBUNDLE (Sigma={metrics.get(p, 0):.4f}): {ig_results[p][:100]}...")

if __name__ == "__main__":
    main()
