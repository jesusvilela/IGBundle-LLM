import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from mfr_utils import construct_phase1_prompt, construct_phase2_prompt

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

def evaluate_arc(model_id, checkpoint_path, split="validation", limit=None, use_mfr=False):
    print(f"Loading Model: {model_id} (Optimized 4-bit via Unsloth)")
    device = "cuda" # Unsloth uses CUDA by default
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = 6144,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    if checkpoint_path:
        print(f"Loading adapter: {checkpoint_path}")
        # Try loading IGBundle weights manually
        ig_weights = os.path.join(checkpoint_path, "adapter_weights.pt")
        if os.path.exists(ig_weights):
            print("Loading IGBundle explicit weights...")
            # We must load to the correct device; unsloth model is on GPU
            model.load_state_dict(torch.load(ig_weights, map_location="cuda"), strict=False)
            
        try:
            # Unsloth compatible Peft loading
            model = PeftModel.from_pretrained(model, checkpoint_path)
            # model = model.merge_and_unload() # Unsloth 4bit doesn't support merge_and_unload easily without upcast
        except Exception as e:
            print(f"Peft load warning: {e}")
            
    # tokenizer = AutoTokenizer.from_pretrained(model_id) # Handled by Unsloth
    
    print("Loading ARC-AGI Dataset (Local)...")
    data_path = "ARC-AGI-master/data/evaluation" # Use evaluation set for testing
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found. Ensure ARC-AGI-master is extracted.")
        return

    tasks = []
    for f in os.listdir(data_path):
        if f.endswith(".json"):
            with open(os.path.join(data_path, f), "r") as json_file:
                tasks.append(json.load(json_file))
    
    # Shuffle or select
    import random
    random.seed(42)
    random.shuffle(tasks)

    if limit:
        tasks = tasks[:limit]
    
    dataset = []
    for t in tasks:
        # Normalize to 'datasets' format for compatibility
        dataset.append(t)
        
    correct = 0
    total = 0
    
    print("Starting Evaluation...")
    for item in tqdm(dataset):
        try:
            prompt, target_grid = format_arc_prompt(item)
            
                
            # --- MFR LOGIC ---
            if use_mfr:
                # Phase 1: Model Construction
                p1 = construct_phase1_prompt(prompt)
                inp1 = tokenizer(p1, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out1_tokens = model.generate(
                        **inp1, 
                        max_new_tokens=512, # Allow space for model definition
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                model_text = tokenizer.decode(out1_tokens[0][inp1.input_ids.shape[1]:], skip_special_tokens=True)
                print(f"\n[MFR Phase 1 Output]:\n{model_text}\n") # Debug
                
                # Phase 2: Reasoning
                final_prompt = construct_phase2_prompt(prompt, model_text)
                inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
            else:
                # Standard CoT/Direct
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=200, 
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            target_str = format_grid(target_grid)
            gen_clean = generated.strip().split("\n")[0].replace(" ", "")
            
            if target_str in gen_clean:
                correct += 1
            
            total += 1
        except Exception as e:
            print(f"Error processing task: {e}")
            continue
        
    print(f"\nResults on ARC-{split}:")
    print(f"Accuracy (Exact String Match): {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to test (ARC is hard!)")
    parser.add_argument("--mfr", action="store_true", help="Enable Model-First Reasoning (2-phase inference)")
    args = parser.parse_args()
    
    evaluate_arc("Qwen/Qwen2.5-7B", args.checkpoint, limit=args.limit, use_mfr=args.mfr)
