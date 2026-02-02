
import os
import glob
import json
import time

def debug_load():
    data_path = os.path.join("data", "ARC-AGI-2", "data", "training")
    print(f"Searching in: {data_path}")
    
    start = time.time()
    json_files = glob.glob(os.path.join(data_path, "*.json"))
    print(f"Glob found {len(json_files)} files in {time.time() - start:.4f}s")
    
    if not json_files:
        print("ERROR: No files found.")
        return

    print("Attempting to load first 100 files...")
    start_load = time.time()
    count = 0
    for i, jf in enumerate(json_files[:100]):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                if 'train' in data:
                    count += 1
        except Exception as e:
            print(f"Failed to load {jf}: {e}")
            
    print(f"Loaded {count}/100 files in {time.time() - start_load:.4f}s")
    
    # Check total size
    total_size = 0
    for jf in json_files:
        total_size += os.path.getsize(jf)
    print(f"Total dataset size: {total_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    debug_load()
