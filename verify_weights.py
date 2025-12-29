import torch
from safetensors import safe_open
import sys
import os

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "output/igbundle_qwen7b/checkpoint-5/adapter_model.safetensors"

if not os.path.exists(checkpoint_path):
    print(f"File not found: {checkpoint_path}")
    sys.exit(1)

print(f"Loading {checkpoint_path}...")
with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
    keys = f.keys()
    adapter_keys = [k for k in keys if "adapter" in k]
    lora_keys = [k for k in keys if "lora" in k]
    
    print(f"Found {len(keys)} total keys.")
    print(f"Found {len(adapter_keys)} adapter keys.")
    print(f"Found {len(lora_keys)} lora keys.")
    
    if len(adapter_keys) > 0:
        print("SUCCESS: IGBundle adapter parameters found in checkpoint.")
        for k in adapter_keys[:5]:
            print(f" - {k}")
    else:
        print("FAILURE: No IGBundle adapter parameters found.")
