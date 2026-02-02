import json
import os
import glob
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import re

# Logic to prevent CUDA OOM
MAX_SEQ_LENGTH = 8192
BATCH_SIZE = 1

def format_grid(grid):
    return str(grid).replace(" ", "")

def create_prompt(task_data):
    prompt = "You are solving an ARC-AGI puzzle. Detect the pattern in the Input grids and generate the correct Output grid.\n\n"
    # Limit to 1-shot to prevent OOM on 8GB VRAM (Text grids are huge)
    if len(task_data['train']) > 0:
        pair = task_data['train'][-1]
        prompt += f"Input:\n{format_grid(pair['input'])}\nOutput:\n{format_grid(pair['output'])}\n\n"
    
    # Test case (ARC usually has 1 test case, sometimes more)
    test_case = task_data['test'][0] 
    prompt += f"Input:\n{format_grid(test_case['input'])}\nOutput:\n"
    return prompt, test_case['output']

def eval_arc(model_path, data_dir, limit=None):
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)

    files = glob.glob(os.path.join(data_dir, "*.json"))
    if limit:
        files = files[:limit]
    
    correct = 0
    total = 0
    valid_syntax = 0
    
    print(f"Running ARC Evaluation on {len(files)} tasks...")
    
    results = []

    for fpath in tqdm(files):
        with open(fpath, 'r') as f:
            data = json.load(f)
        
        prompt, ground_truth = create_prompt(data)
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Determine strict generation length to prevent hang
        # ARC grids are small, usually < 30x30. Token count won't be massive.
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["\n\n", "Input:", "Output:"], 
            tokenizer=tokenizer
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the new part
        prediction_text = generated_text[len(prompt):].strip()
        
        # Normalize
        prediction_text = prediction_text.split("Input:")[0].strip()
        
        # Try parse
        is_correct = False
        is_valid = False
        try:
            # Safe eval of list structure
            pred_grid = json.loads(prediction_text)
            if isinstance(pred_grid, list) and isinstance(pred_grid[0], list):
                is_valid = True
                if pred_grid == ground_truth:
                    is_correct = True
                    correct += 1
        except:
            pass
            
        if is_valid:
            valid_syntax += 1

        total += 1
        results.append({
            "file": os.path.basename(fpath),
            "correct": is_correct,
            "valid_syntax": is_valid,
            "prediction": prediction_text,
            "truth": str(ground_truth)
        })

    print("-" * 30)
    print(f"Total: {total}")
    print(f"Correct (Exact Match): {correct} ({correct/total*100:.2f}%)")
    print(f"Valid Parsing: {valid_syntax} ({valid_syntax/total*100:.2f}%)")
    
    with open("arc_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    print("Starting script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to run (default 20)")
    args = parser.parse_args()
    print(f"Arguments parsed. Limit: {args.limit}")
    
    try:
        eval_arc("output/igbundle_qwen7b_riemannian_merged", "data/arc_agi", args.limit)
    except Exception as e:
        import traceback
        traceback.print_exc()
