
import torch
import os

ckpt_path = "output/igbundle_qwen7b/checkpoint-600/adapter_weights.pt"
print(f"Inspecting {ckpt_path}")

if os.path.exists(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    for k, v in list(state_dict.items())[:5]: # just first 5 to see structure
        print(f"{k}: {v.shape}")
        
    # Check specific keys
    for k, v in state_dict.items():
        if "input_proj" in k or "bottleneck" in k:
            print(f"SPECIFIC {k}: {v.shape}")
            break
else:
    print("Checkpoint not found!")
