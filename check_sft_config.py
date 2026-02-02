
from trl import SFTConfig
import inspect

try:
    sig = inspect.signature(SFTConfig.__init__)
    print("SFTConfig Signature:")
    print(sig)
except Exception as e:
    print(f"Error: {e}")
