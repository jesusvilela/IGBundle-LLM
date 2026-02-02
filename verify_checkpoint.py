
import torch
import os

checkpoint_dir = "output/igbundle_qwen7b_resized/checkpoint-600"
files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]

print(f"Checking files in {checkpoint_dir}...")

for f in files:
    path = os.path.join(checkpoint_dir, f)
    print(f"\nLoading {f}...")
    try:
        state = torch.load(path, map_location="cpu")
        print(f"Keys: {len(state)}")
        
        all_bf16 = True
        for k, v in state.items():
            if torch.is_tensor(v):
                if v.dtype != torch.bfloat16:
                    print(f"  ❌ {k}: {v.dtype} (Expected bfloat16)")
                    all_bf16 = False
                # else:
                #     print(f"  ✅ {k}: {v.dtype}")
        
        if all_bf16:
            print("  ✅ All tensors are BFloat16.")
        else:
            print("  ❌ Some tensors are NOT BFloat16.")
            
    except Exception as e:
        print(f"  ❌ Error loading {f}: {e}")
