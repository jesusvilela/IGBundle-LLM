
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
else:
    print("No CUDA")
