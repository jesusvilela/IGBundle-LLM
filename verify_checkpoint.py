import torch
import os
from safetensors.torch import load_file

def check_checkpoint(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("Path does not exist.")
        return
        
    # Check for .safetensors
    st_path = os.path.join(path, "adapter_model.safetensors")
    
    # Check for IGBundle custom weights
    ig_path = os.path.join(path, "adapter_weights.pt")
    if os.path.exists(ig_path):
        print(f"[OK] Found IGBundle weights: {ig_path}")
        ig_weights = torch.load(ig_path, map_location="cpu")
        # Check IG weights for NaNs
        for k, v in ig_weights.items():
            if torch.isnan(v).any():
                print(f"NAN found in IGBundle weight: {k}")
                has_nan = True
            if torch.isinf(v).any():
                print(f"INF found in IGBundle weight: {k}")
                has_nan = True
    else:
        print("[WARNING] IGBundle weights (adapter_weights.pt) NOT found!")

    if os.path.exists(st_path):
        state_dict = load_file(st_path)
    else:
        # Fallback to .bin
        bin_path = os.path.join(path, "adapter_model.bin")
        if not os.path.exists(bin_path):
            print("No adapter_model found.")
            return
        state_dict = torch.load(bin_path, map_location="cpu")
        
    has_nan = False
    for k, v in state_dict.items():
        if torch.isnan(v).any():
            print(f"NAN found in {k}")
            has_nan = True
        if torch.isinf(v).any():
            print(f"INF found in {k}")
            has_nan = True
            
    if not has_nan:
        print("Checkpoint is CLEAN (no NaNs or Infs).")
    else:
        print("Checkpoint is CORRUPT (NaNs or Infs found).")

if __name__ == "__main__":
    check_checkpoint("h:/LLM-MANIFOLD/igbundle-llm/output/igbundle_qwen7b/checkpoint-1")
