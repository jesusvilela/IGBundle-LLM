import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

def format_grid(grid):
    return str(grid).replace(" ", "")

def format_arc_prompt(example, num_shots=1):
    # ARC tasks have "train" (examples) and "test" (question)
    # We use the 'train' pairs as few-shot examples
    prompt = "The following logical reasoning puzzle involves transforming input grids to output grids.\n\n"
    
    # Few-shot examples from the task itself
    train_pairs = example['train']
    for i in range(min(num_shots, len(train_pairs))):
        inp = train_pairs[i]['input']
        out = train_pairs[i]['output']
        prompt += f"Example {i+1}:\nInput: {format_grid(inp)}\nOutput: {format_grid(out)}\n\n"
        
    # The actual test case
    test_inp = example['test'][0]['input'] # Usually only 1 test case per task in validation
    prompt += f"Test:\nInput: {format_grid(test_inp)}\nOutput:"
    
    return prompt, example['test'][0]['output']

def evaluate_arc(model_id, checkpoint_path, split="validation", limit=None):
    print(f"Loading Model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if checkpoint_path:
        print(f"Loading adapter: {checkpoint_path}")
        # Try loading IGBundle weights manually if standard Peft fails or to ensure coverage
        ig_weights = os.path.join(checkpoint_path, "adapter_weights.pt")
        if os.path.exists(ig_weights):
            print("Loading IGBundle explicit weights...")
            model.load_state_dict(torch.load(ig_weights, map_location=device), strict=False)
            
        try:
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model = model.merge_and_unload() # Merge for speed
        except Exception as e:
            print(f"Peft load warning: {e}")
            
    model.eval()
    
    print("Loading ARC-AGI Dataset (huggingface: chollet/ARC)...")
    try:
        # Note: 'chollet/ARC' might not be directly hosting the data easily for 'load_dataset' without config
        # We'll use a known community mirror if official fails, e.g. 'fchollet/ARC' doesn't exist as HF dataset directly often
        # Using a reliable mirror: 'barc0/arc_puzzles' or similar. 
        # Actually, let's use a local loader or a trustworthy mirror. 
        # 'giganticode/ARC' is a common one.
        dataset = load_dataset("giganticode/ARC", split=split)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    if limit:
        dataset = dataset.select(range(limit))
        
    correct = 0
    total = 0
    
    print("Starting Evaluation...")
    for item in tqdm(dataset):
        # The structure of 'item' depends on the specific HF dataset version.
        # giganticode/ARC details:
        # features: ['id', 'train', 'test']
        
        prompt, target_grid = format_arc_prompt(item)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                do_sample=False, # Greco search/Grid search determinism
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Simple parsing: Look for the grid format
        # This is very fragile. Validating ARC requires parsing the 2D array.
        target_str = format_grid(target_grid)
        
        # Clean generation
        gen_clean = generated.strip().split("\n")[0] # Take first line
        gen_clean = gen_clean.replace(" ", "")
        
        if target_str in gen_clean:
            correct += 1
        
        total += 1
        
    print(f"\nResults on ARC-{split}:")
    print(f"Accuracy (Exact String Match): {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to test (ARC is hard!)")
    args = parser.parse_args()
    
    evaluate_arc("Qwen/Qwen2.5-7B", args.checkpoint, limit=args.limit)
