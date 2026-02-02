import shutil
import os

cache_dir = os.path.expanduser("~/.cache/lm_eval")
print(f"Targeting cache dir: {cache_dir}")

if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print("Successfully removed lm_eval cache.")
    except Exception as e:
        print(f"Error removing cache: {e}")
else:
    print("Cache directory not found.")
