import argparse
import json
import os
import torch
import math
import re
from tqdm import tqdm
from unsloth import FastLanguageModel

def format_grid(grid):
    return str(grid).replace(" ", "")

def format_arc_prompt(example, num_shots=1):
    prompt = "The following logical reasoning puzzle involves transforming input grids to output grids.\n\n"
    
    # Few-shot examples from the task itself
    train_pairs = example['train']
    for i in range(min(num_shots, len(train_pairs))):
        inp = train_pairs[i]['input']
        out = train_pairs[i]['output']
        prompt += f"Example {i+1}:\nInput: {format_grid(inp)}\nOutput: {format_grid(out)}\n\n"
        
    test_inp = example['test'][0]['input']
    prompt += f"Test:\nInput: {format_grid(test_inp)}\nOutput:"
    
    return prompt, example['test'][0]['output']

def calculate_confidence_interval(k, n, confidence=0.95):
    if n == 0: return 0, 0
    z = 1.96 # Approx for 95%
    p_hat = k / n
    numerator = p_hat + z*z/(2*n) + z * math.sqrt((p_hat*(1-p_hat)/n) + z*z/(4*n*n))
    denominator = 1 + z*z/n
    upper = numerator / denominator
    low_numerator = p_hat + z*z/(2*n) - z * math.sqrt((p_hat*(1-p_hat)/n) + z*z/(4*n*n))
    lower = low_numerator / denominator
    return lower, upper

def extract_grid(text):
    pattern = r'\[\s*\[[\d\s,]*\](?:\s*,\s*\[[\d\s,]*\])*\s*\]'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].replace("\n", "").replace(" ", "")
    
    pattern_loose = r'\[\s*\[.*?\]\s*\]'
    matches_loose = re.findall(pattern_loose, text, re.DOTALL)
    for m in reversed(matches_loose):
        m_clean = m.replace(" ", "").replace("\n", "")
        if m_clean.count(",") > 5 and any(c.isdigit() for c in m_clean):
            if "for" not in m_clean and "range" not in m_clean:
                return m_clean
    return ""

def load_tasks(data_path):
    tasks = []
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found.")
        return tasks
    for f in os.listdir(data_path):
        if f.endswith(".json"):
            with open(os.path.join(data_path, f), "r") as json_file:
                tasks.append({"id": f[:-5], "data": json.load(json_file)})
    # Sort for determinism
    tasks.sort(key=lambda x: x["id"])
    return tasks

def evaluate_arc(model, tokenizer, tasks, device="cuda"):
    correct = 0
    total = 0
    results_log = []
    
    for item_wrapper in tqdm(tasks):
        item = item_wrapper["data"]
        task_id = item_wrapper["id"]
        result_entry = {"task_id": task_id, "correct": False}
        
        try:
            prompt, target_grid = format_arc_prompt(item)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=2048,
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result_entry["generated"] = generated
            result_entry["target"] = format_grid(target_grid)
            
            target_clean = format_grid(target_grid).replace(" ", "")
            gen_clean = extract_grid(generated)
            
            if gen_clean == target_clean or target_clean in gen_clean:
                correct += 1
                result_entry["correct"] = True
            
            total += 1
            results_log.append(result_entry)
            
        except Exception as e:
            result_entry["error"] = str(e)
            results_log.append(result_entry)
            
    lower, upper = calculate_confidence_interval(correct, total)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "confidence_interval": [lower, upper],
        "details": results_log
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--checkpoint", default=None, help="Path to PEFT adapter")
    parser.add_argument("--limit", type=int, default=None, help="Number of tasks to evaluate")
    parser.add_argument("--output", default="falsification_arc_results.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}, checkpoint: {args.checkpoint}")
    model_name = args.checkpoint if args.checkpoint else args.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        load_in_4bit=True,
        device_map={"": 0}
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading ARC-AGI v1 Tasks...")
    tasks_v1 = load_tasks("ARC-AGI-master/data/evaluation")
    print("Loading ARC-AGI v2 Tasks...")
    tasks_v2 = load_tasks("data/ARC-AGI-2/data/evaluation")

    if args.limit:
        tasks_v1 = tasks_v1[:args.limit]
        tasks_v2 = tasks_v2[:args.limit]

    print(f"Evaluating ARC-AGI v1 ({len(tasks_v1)} tasks)...")
    results_v1 = evaluate_arc(model, tokenizer, tasks_v1)
    print(f"ARC-AGI v1 Results: {results_v1['accuracy']*100:.2f}% ({results_v1['correct']}/{results_v1['total']})")

    print(f"Evaluating ARC-AGI v2 ({len(tasks_v2)} tasks)...")
    results_v2 = evaluate_arc(model, tokenizer, tasks_v2)
    print(f"ARC-AGI v2 Results: {results_v2['accuracy']*100:.2f}% ({results_v2['correct']}/{results_v2['total']})")

    final_output = {
        "config": "Config D (Bare) or PEFT",
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "arc_agi_v1": {
            "accuracy": results_v1["accuracy"],
            "confidence_interval": results_v1["confidence_interval"],
            "correct": results_v1["correct"],
            "total": results_v1["total"],
            "details": results_v1["details"]
        },
        "arc_agi_v2": {
            "accuracy": results_v2["accuracy"],
            "confidence_interval": results_v2["confidence_interval"],
            "correct": results_v2["correct"],
            "total": results_v2["total"],
            "details": results_v2["details"]
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
