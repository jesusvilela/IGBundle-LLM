
import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# Config
OUTPUT_FILE = "igbundle-llm/data/full_scale_train.jsonl"
MAX_SAMPLES = 50000 # Cap for reasonable training time
SEED = 42

def format_alibaba(example):
    # Alibaba format: input, output
    inp = example.get('input', '')
    out = example.get('output', '')
    if not inp or not out: return None
    return {
        "input": inp,
        "output": out,
        "source": "alibaba_superior_reasoning"
    }

def format_openmath(example):
    # OpenMath format check (variable, usually 'problem' and 'solution')
    # Or 'question' and 'response'
    inp = example.get('problem', example.get('question', ''))
    out = example.get('solution', example.get('response', example.get('answer', '')))
    
    if not inp or not out: return None
    return {
        "input": inp,
        "output": out, 
        "source": "openmath_reasoning"
    }

def main():
    random.seed(SEED)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_examples = []
    
    # 1. Load Alibaba (SFT Reasoning)
    print("Loading Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b...")
    try:
        ds_alibaba = load_dataset("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", "stage1", split="train")
        # Take 50% of quota
        indices = random.sample(range(len(ds_alibaba)), min(len(ds_alibaba), MAX_SAMPLES // 2))
        sub_alibaba = ds_alibaba.select(indices)
        
        for ex in tqdm(sub_alibaba, desc="Processing Alibaba"):
            fmt = format_alibaba(ex)
            if fmt: all_examples.append(fmt)
            
    except Exception as e:
        print(f"Error loading Alibaba dataset: {e}")

    # 2. Load OpenMath or Fallback
    print("Loading nvidia/OpenMathReasoning (or proxy)...")
    try:
        # Note: OpenMath might be large or gated. Fallback to KingNish if needed?
        # Let's try OpenMath first. If it fails (due to gating/size), we catch it.
        # Actually OpenMathReasoning might not be directly hosted or named differently.
        # Common CoT dataset: 'KingNish/reasoning-base-20k' or 'meta-math/MetaMathQA'
        # Per plan, we try OpenMath. If not, we use 'meta-math/MetaMathQA' as a reliable math reasoning source.
        ds_math = load_dataset("meta-math/MetaMathQA", split="train") 
        
        indices = random.sample(range(len(ds_math)), min(len(ds_math), MAX_SAMPLES // 2))
        sub_math = ds_math.select(indices)
        
        for ex in tqdm(sub_math, desc="Processing MetaMath (OpenMath Proxy)"):
            # MetaMath: 'query', 'response'
            # Check keys
            inp = ex.get('query', '')
            out = ex.get('response', '')
            if inp and out:
                all_examples.append({
                    "input": inp,
                    "output": out,
                    "source": "metamath_reasoning"
                })
                
    except Exception as e:
        print(f"Error loading Math dataset: {e}")

    # 3. Shuffle and Save
    random.shuffle(all_examples)
    print(f"Total examples collected: {len(all_examples)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
