import torch
from safetensors.torch import load_file
import statistics

def inspect_adapter():
    path = "output/igbundle_qwen7b_riemannian/checkpoint-100/adapter_model.safetensors"
    print(f"Loading adapter from {path}...")
    try:
        tensors = load_file(path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Found {len(tensors)} tensors.")
    
    abs_means = []
    max_vals = []
    
    for name, tensor in tensors.items():
        if "lora" in name or "adapter" in name:
            val = tensor.float().abs()
            mean = val.mean().item()
            mx = val.max().item()
            abs_means.append(mean)
            max_vals.append(mx)
            # print(f"{name}: mean_abs={mean:.6f}, max={mx:.6f}")

    if abs_means:
        avg_weight = sum(abs_means) / len(abs_means)
        avg_max = sum(max_vals) / len(max_vals)
        print(f"\nGlobal Statistics:")
        print(f"  Average Absolute Weight: {avg_weight:.6f}")
        print(f"  Average Max Weight:      {avg_max:.6f}")
        
        if avg_weight < 1e-6:
            print("  [WARNING] Weights are extremely small. Training might have failed.")
        else:
            print("  [OK] Weights appear non-zero.")
    else:
        print("No adapter tensors found.")

if __name__ == "__main__":
    inspect_adapter()
