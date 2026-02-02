
import os
import json
from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
OUTPUT_DIR = "igbundle-llm/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "deepplanning_train.jsonl")

def format_example(example):
    # Alibaba format: input, output, domain
    inp = example.get('input', '')
    out = example.get('output', '')
    
    if not inp or not out:
        return None
        
    return {
        "input": inp,
        "output": out,
        "system": "You are a helpful assistant capable of deep reasoning."
    }

def main():
    print(f"Downloading {DATASET_ID} (stage1)...")
    try:
        ds = load_dataset(DATASET_ID, "stage1", split="train")
        # Take a subset for pilot if too large
        ds = ds.select(range(min(len(ds), 2000))) 
    except Exception as e:
        print(f"Error loading: {e}")
        return

    print(f"Processing {len(ds)} examples...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    valid_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in tqdm(ds):
            formatted = format_example(ex)
            if formatted:
                f.write(json.dumps(formatted) + "\n")
                valid_count += 1
                
    print(f"Saved {valid_count} examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
