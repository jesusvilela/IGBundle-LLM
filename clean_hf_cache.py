
import shutil
import os
from transformers.utils import TRANSFORMERS_CACHE

# Target specific model
model_id = "models--unsloth--Qwen2.5-7B-Instruct"
target_dir = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), model_id)

print(f"Targeting cache dir: {target_dir}")

if os.path.exists(target_dir):
    try:
        shutil.rmtree(target_dir)
        print(f"Successfully removed {target_dir}")
    except Exception as e:
        print(f"Error removing cache: {e}")
else:
    print("Cache directory not found. Already clean?")
