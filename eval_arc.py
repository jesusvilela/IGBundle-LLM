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

def calculate_confidence_interval(k, n, confidence=0.95):
    """
    Calculate Wilson Score Interval for binomial proportion.
    """
    if n == 0: return 0, 0
    import math
    z = 1.96 # Approx for 95%
    p_hat = k / n
    numerator = p_hat + z*z/(2*n) + z * math.sqrt((p_hat*(1-p_hat)/n) + z*z/(4*n*n))
    denominator = 1 + z*z/n
    upper = numerator / denominator
    low_numerator = p_hat + z*z/(2*n) - z * math.sqrt((p_hat*(1-p_hat)/n) + z*z/(4*n*n))
    lower = low_numerator / denominator
    return lower, upper

def evaluate_arc(model_id, checkpoint_path, split="validation", limit=None, use_mfr=False):
    print(f"Loading Model: {model_id} (Optimized 4-bit via Unsloth)")
    print("Scientific Evaluation Mode: ACTIVE")
    device = "cuda"
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
        ig_weights = os.path.join(checkpoint_path, "adapter_weights.pt")
        if os.path.exists(ig_weights):
            print("Loading IGBundle explicit weights...")
            model.load_state_dict(torch.load(ig_weights, map_location="cuda"), strict=False)
            
        try:
            model = PeftModel.from_pretrained(model, checkpoint_path)
        except Exception as e:
            print(f"Peft load warning: {e}")
    
    print("Loading ARC-AGI Dataset (Local)...")
    data_path = "ARC-AGI-master/data/evaluation"
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return

    tasks = []
    for f in os.listdir(data_path):
        if f.endswith(".json"):
            with open(os.path.join(data_path, f), "r") as json_file:
                tasks.append({"id": f[:-5], "data": json.load(json_file)})
    
    import random
    random.seed(42)
    torch.manual_seed(42)
    random.shuffle(tasks)

    if limit:
        tasks = tasks[:limit]
    
    correct = 0
    total = 0
    mfr_compliant_count = 0
    results_log = []
    
    print(f"Starting Evaluation on {len(tasks)} tasks...")
    for item_wrapper in tqdm(tasks):
        item = item_wrapper["data"]
        task_id = item_wrapper["id"]
        result_entry = {"task_id": task_id, "correct": False, "mfr_compliant": False}
        
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
                        max_new_tokens=1024, # Increased for robust model definition
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                model_text = tokenizer.decode(out1_tokens[0][inp1.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Verify MFR format
                if "## PROBLEM MODEL" in model_text or "ENTITIES" in model_text:
                    mfr_compliant_count += 1
                    result_entry["mfr_compliant"] = True
                
                # Phase 2: Reasoning
                final_prompt = construct_phase2_prompt(prompt, model_text)
                inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=2048, # Increased for complex grids
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result_entry["generated"] = generated
            
            target_str = format_grid(target_grid)
            gen_clean = generated.strip().split("\n")[0].replace(" ", "")
            
            # Robust check: look for target grid in last few lines
            if target_str in gen_clean or target_str in generated.replace(" ", "")[-len(target_str)*2:]:
                correct += 1
                result_entry["correct"] = True
            
            total += 1
            results_log.append(result_entry)
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            result_entry["error"] = str(e)
            results_log.append(result_entry)
            continue
        
    lower, upper = calculate_confidence_interval(correct, total)
    
    print("\n" + "="*40)
    print(f"SCIENTIFIC EVALUATION REPORT (n={total})")
    print(f"Model: {model_id} + {checkpoint_path}")
    print(f"Mode: {'MFR (Model-First Reasoning)' if use_mfr else 'Standard CoT'}")
    print("-" * 40)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"95% Confidence Interval: [{lower*100:.2f}%, {upper*100:.2f}%]")
    if use_mfr:
        print(f"MFR Compliance Rate: {mfr_compliant_count}/{total} ({mfr_compliant_count/total*100:.1f}%)")
    print("="*40)
    
    # Save detailed logs
    with open("arc_evaluation_results.json", "w") as f:
        json.dump({
            "summary": {
                "accuracy": correct/total if total > 0 else 0,
                "confidence_interval": [lower, upper],
                "mfr_compliance": mfr_compliant_count/total if total > 0 else 0,
                "total": total
            },
            "details": results_log
        }, f, indent=2)
    print("Detailed results saved to 'arc_evaluation_results.json'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to test")
    parser.add_argument("--mfr", action="store_true", help="Enable Model-First Reasoning")
    args = parser.parse_args()
    
    evaluate_arc("Qwen/Qwen2.5-7B", args.checkpoint, limit=args.limit, use_mfr=args.mfr)
