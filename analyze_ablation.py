import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import yaml
import sys

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from igbundle.integrations.hf_patch import wrap_hf_candidate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Simple Config Object
class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def measure_checkpoint(checkpoint_dir, config_path):
    print(f"--- Measuring {checkpoint_dir} ---")
    adapter_weights_path = os.path.join(checkpoint_dir, "adapter_weights.pt")
    
    if not os.path.exists(adapter_weights_path):
        print(f"Error: {adapter_weights_path} not found.")
        return None

    # Load Config
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(cfg_dict)

    # Load Base Model
    # Use 4-bit loading for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return None

    # Inject IGBundle Adapter
    # We need to set hidden_size correctly
    if hasattr(model.config, "hidden_size"):
        cfg.ig_adapter.hidden_size = model.config.hidden_size
        
    model = wrap_hf_candidate(model, cfg.ig_adapter)
    
    # Load Weights
    print(f"Loading adapter weights from {adapter_weights_path}...")
    state_dict = torch.load(adapter_weights_path)
    # The keys in state_dict likely match the module names if saved via model.state_dict() or specific modules
    # In train.py, manual save might have saved only IGBundle keys.
    # Let's try loading with strict=False
    keys = model.load_state_dict(state_dict, strict=False)
    # Check if igbundle keys were loaded
    missing_ig = [k for k in keys.missing_keys if "igbundle" in k]
    if missing_ig:
        print(f"Warning: Missing IGBundle keys: {missing_ig[:5]}...")
    else:
        print("IGBundle weights loaded successfully.")
    
    # Hooks for Analysis
    captured_data = {
        "entropies": [],
        "spreads": [],
        "norms": []
    }
    
    def hook_fn(module, input, output):
        # output is (hidden, state) tuple
        if isinstance(output, tuple):
            _, state = output
            # state has w_logits (B,T,P), m (B,T,P,D), u (B,T,P,K)
            
            # 1. Entropy of mixing weights
            w = F.softmax(state.w_logits, dim=-1) # (B,T,P)
            entropy = -torch.sum(w * torch.log(w + 1e-9), dim=-1).mean().item()
            captured_data["entropies"].append(entropy)
            
            # 2. Spread of components (Distances between m_i)
            # m: (B,T,P,D)
            avg_m = state.m.mean(dim=[0,1]) # (P, D)
            # Pairwise distances
            pdist = torch.norm(avg_m.unsqueeze(0) - avg_m.unsqueeze(1), dim=-1)
            spread = pdist.mean().item()
            captured_data["spreads"].append(spread)
            
            # 3. Norms (Hyperbolic boundary check)
            # In ops.py, means_hyp = tanh(means). state.m is likely 'means'.
            # We want to check if the learned 'means' are pushing 0 (linear) or large (boundary)
            # If tanh(means) -> 1, then means -> large.
            # So norm of m is a decent proxy for saturation.
            norms = torch.norm(avg_m, dim=-1).mean().item()
            captured_data["norms"].append(norms)

    # Register hooks
    handles = []
    for n, m in model.named_modules():
        if "IGBundleAdapter" in m.__class__.__name__:
            handles.append(m.register_forward_hook(hook_fn))
            
    if not handles:
        print("Warning: No IGBundleAdapter modules found to hook.")
        
    # Run Inference
    inputs = tokenizer(["The quick brown fox jumps over the lazy dog." * 2], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
        
    for h in handles:
        h.remove()
        
    results = {
        "entropy": np.mean(captured_data["entropies"]) if captured_data["entropies"] else 0,
        "spread": np.mean(captured_data["spreads"]) if captured_data["spreads"] else 0,
        "norm": np.mean(captured_data["norms"]) if captured_data["norms"] else 0
    }
    
    print(f"Results: {results}")
    return results

if __name__ == "__main__":
    R_dir = "output/igbundle_qwen7b_riemannian/checkpoint-25"
    R_conf = "configs/qwen25_7b_igbundle_riemannian.yaml"
    
    E_dir = "output/igbundle_qwen7b_euclidean/checkpoint-25"
    E_conf = "configs/qwen25_7b_igbundle_euclidean.yaml"
    
    import gc
    
    print("ANALYSIS: Riemannian vs Euclidean Inductive Bias")
    
    r_res = measure_checkpoint(R_dir, R_conf)
    
    # Cleanup to free VRAM for next model load
    gc.collect()
    torch.cuda.empty_cache()
    
    e_res = measure_checkpoint(E_dir, E_conf)
        
    if r_res and e_res:
        print("\n--- COMPARISON ---")
        print(f"metric        | Riemannian | Euclidean | Delta")
        print(f"--------------|------------|-----------|-------")
        print(f"Entropy (H)   | {r_res['entropy']:.4f}     | {e_res['entropy']:.4f}    | {r_res['entropy']-e_res['entropy']:.4f}")
        print(f"Spread (Dist) | {r_res['spread']:.4f}     | {e_res['spread']:.4f}    | {r_res['spread']-e_res['spread']:.4f}")
        print(f"Norm (|x|)    | {r_res['norm']:.4f}     | {e_res['norm']:.4f}    | {r_res['norm']-e_res['norm']:.4f}")
        
        print("\nInterpretation:")
        if r_res['entropy'] < e_res['entropy']:
            print(">> Riemannian model shows LOWER entropy, indicating sharper specialization (modes are more distinct).")
        else:
            print(">> Riemannian model shows HIGHER entropy, suggesting more blended representations.")
            
        if r_res['spread'] > e_res['spread']:
             print(">> Riemannian components are MORE spread out (Euclidean), suggesting better utilization of the latent volume.")
             
        if r_res['norm'] > e_res['norm']:
             print(">> Riemannian components have LARGER norms, pushing towards the Poincaré boundary (Hyperbolic/Tree-like behavior).")
        else:
             print(">> Riemannian components have SMALLER norms, staying in the linear (Euclidean) regime of the ball.")
