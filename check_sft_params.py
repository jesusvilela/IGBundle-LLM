
from trl import SFTTrainer
import inspect

try:
    sig = inspect.signature(SFTTrainer.__init__)
    print("SFTTrainer Signature:")
    for name, param in sig.parameters.items():
        if 'token' in name or 'process' in name:
            print(f"{name}: {param.annotation}")
except Exception as e:
    print(f"Error: {e}")
