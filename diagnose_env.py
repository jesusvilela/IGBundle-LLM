"""
Environment Diagnostic Script for IGBundle LLM Training
Run this to verify your environment is correctly configured.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("IGBundle Environment Diagnostic")
print("=" * 60)

# 1. Python Version Check
print(f"\n[1] Python Version: {sys.version}")
if sys.version_info >= (3, 13):
    print("   ⚠️  WARNING: Python 3.13+ detected - may have compatibility issues")
    print("   ➡️  Recommended: Python 3.10 or 3.11")
elif sys.version_info >= (3, 10):
    print("   ✅ Python version OK")
else:
    print("   ❌ Python version too old - need 3.10+")

# 2. Apply Triton fix BEFORE importing torch
print("\n[2] Applying Triton compatibility fix...")
try:
    from igbundle.utils import triton_fix
    print("   ✅ Triton fix applied")
except Exception as e:
    print(f"   ❌ Triton fix failed: {e}")

# 3. Block TorchAO if problematic
print("\n[3] Checking TorchAO...")
os.environ['DISABLE_TORCHAO'] = '1'  # Disable to prevent issues
try:
    # Try importing torch first
    import torch
    print(f"   ✅ PyTorch {torch.__version__} loaded")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")
    sys.exit(1)

# 4. Check transformers
print("\n[4] Checking Transformers...")
try:
    import transformers
    print(f"   ✅ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"   ❌ Transformers import failed: {e}")

# 5. Check PEFT
print("\n[5] Checking PEFT...")
try:
    import peft
    print(f"   ✅ PEFT {peft.__version__}")
except ImportError as e:
    print(f"   ❌ PEFT import failed: {e}")

# 6. Check bitsandbytes
print("\n[6] Checking bitsandbytes...")
try:
    import bitsandbytes as bnb
    print(f"   ✅ bitsandbytes loaded")
except ImportError as e:
    print(f"   ⚠️  bitsandbytes not available: {e}")
    print("   ➡️  4-bit quantization will be disabled")

# 7. Check accelerate
print("\n[7] Checking Accelerate...")
try:
    import accelerate
    print(f"   ✅ Accelerate {accelerate.__version__}")
except ImportError as e:
    print(f"   ❌ Accelerate import failed: {e}")

# 8. Test IGBundle modules
print("\n[8] Testing IGBundle modules...")
try:
    from igbundle.modules.adapter import IGBundleAdapter
    from igbundle.integrations.hf_patch import IGBundleBlockWrapper, wrap_hf_candidate
    print("   ✅ IGBundle modules loaded")
except ImportError as e:
    print(f"   ❌ IGBundle import failed: {e}")

# 9. Quick GPU memory test
print("\n[9] GPU Memory Test...")
if torch.cuda.is_available():
    try:
        # Allocate a small tensor
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"   ✅ GPU compute works")
        print(f"   ✅ Free GPU memory: {free_mem / 1e9:.2f} GB")
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
else:
    print("   ⚠️  No CUDA - will use CPU (very slow)")

# 10. Summary
print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)

# Check for critical issues
issues = []
if sys.version_info >= (3, 13):
    issues.append("Python 3.13 may cause issues - consider Python 3.11")
if not torch.cuda.is_available():
    issues.append("No CUDA - training will be extremely slow")

if issues:
    print("\n⚠️  Issues Found:")
    for issue in issues:
        print(f"   • {issue}")
else:
    print("\n✅ Environment looks good! Try running:")
    print("   python train.py --smoke_test")
