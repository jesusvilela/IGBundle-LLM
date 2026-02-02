
from trl import SFTTrainer
import inspect

try:
    sig = inspect.signature(SFTTrainer.__init__)
    print("SFTTrainer Signature:")
    print(sig)
except Exception as e:
    print(f"Error: {e}")
