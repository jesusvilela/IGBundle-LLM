import traceback
import sys
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    import unsloth
    print("Unsloth import OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
