
import torch
import os
import shutil

src_path = "output/igbundle_qwen7b/checkpoint-600"
dst_path = "output/igbundle_qwen7b/checkpoint-600-small"
adapter_file = "adapter_weights.pt"

print(f"Resizing adapter from {src_path} to {dst_path}")

os.makedirs(dst_path, exist_ok=True)

# Copy all files first
for item in os.listdir(src_path):
    s = os.path.join(src_path, item)
    d = os.path.join(dst_path, item)
    if os.path.isfile(s) and item != adapter_file:
        shutil.copy2(s, d)

# Load weights
weights = torch.load(os.path.join(src_path, adapter_file), map_location="cpu")
new_weights = {}

# Target dimensions
# prev: D_bot=256, D=128
# new : D_bot=128, D=64

DIM_MAP = {
    # Key pattern substring : (dim_index, slice_size)
    "input_proj.weight": (0, 128), # [256, 3584] -> [128, 3584]
    "input_proj.bias": (0, 128),   # [256] -> [128]
    "output_proj.weight": (1, 128), # [3584, 256] -> [3584, 128]
    # Internal fiber dims?
    # proj_w, proj_m etc might depend on D_bot as input
    # Let's inspect based on shape matching
}

def resize_tensor(tensor, name):
    # Heuristic resizing based on shape match
    # D_bot transitions: 256 -> 128
    # D transitions: 128 -> 64
    
    sh = tensor.shape
    new_t = tensor.clone()
    
    # Dimension 0
    if sh[0] == 256:
        new_t = new_t[:128]
    elif sh[0] == 128:
        new_t = new_t[:64]
    elif sh[0] == 512:
        new_t = new_t[:256]
        
    # Dimension 1
    if len(sh) > 1:
        if sh[1] == 256:
            # Check if this is [128, 256] -> we might have already sliced dim0 to 64
            # We want to slice dim1 to 128
            new_t = new_t[:, :128]
        elif sh[1] == 128:
            new_t = new_t[:, :64]
            
    return new_t.to(torch.bfloat16)

print("Processing tensors...")
for k, v in weights.items():
    v_new = resize_tensor(v, k)
    if v_new.shape != v.shape:
        print(f"Resized {k}: {v.shape} -> {v_new.shape}")
    new_weights[k] = v_new

torch.save(new_weights, os.path.join(dst_path, adapter_file))
print("✅ Done. Saved to", dst_path)
