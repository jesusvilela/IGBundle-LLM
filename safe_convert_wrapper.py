
import sys
import os

# === CRITICAL: Apply compatibility fixes BEFORE any imports ===
os.environ['DISABLE_TORCHAO'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
sys.modules['torchao'] = None # Force ImportError

# Add local src to path for triton_fix if needed
_src_path = os.path.join(os.path.dirname(__file__), "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Apply Triton fixes
try:
    from igbundle.utils import triton_fix
except ImportError:
    print("Warning: Could not import igbundle.utils.triton_fix")

# Add llama.cpp to path
llama_cpp_path = os.path.join(os.path.dirname(__file__), "llama.cpp")
if llama_cpp_path not in sys.path:
    sys.path.insert(0, llama_cpp_path)

print("🚀 Launching Safe GGUF Conversion...")
import convert_hf_to_gguf

if __name__ == "__main__":
    convert_hf_to_gguf.main()
