
import argparse
import json
import os
from llama_cpp import Llama
from datasets import load_dataset
from tqdm import tqdm
from mfr_utils import construct_phase1_prompt, construct_phase2_prompt

def format_grid(grid):
    return str(grid).replace(" ", "")

def format_arc_prompt(example, num_shots=1):
    prompt = "The following logical reasoning puzzle involves transforming input grids to output grids.\n\n"
    train_pairs = example['train']
    for i in range(min(num_shots, len(train_pairs))):
        inp = train_pairs[i]['input']
        out = train_pairs[i]['output']
        prompt += f"Example {i+1}:\nInput: {format_grid(inp)}\nOutput: {format_grid(out)}\n\n"
    test_inp = example['test'][0]['input']
    prompt += f"Test:\nInput: {format_grid(test_inp)}\nOutput:"
    return prompt, example['test'][0]['output']

def evaluate_arc_gguf(model_path, data_path="ARC-AGI-master/data/evaluation", limit=None, use_mfr=False):
    print(f"Loading GGUF Model: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=6144, n_gpu_layers=-1) # -1 for all layers on GPU

    tasks = []
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} not found.")
        return

    for f in os.listdir(data_path):
        if f.endswith(".json"):
            with open(os.path.join(data_path, f), "r") as json_file:
                tasks.append({"id": f[:-5], "data": json.load(json_file)})
    
    if limit:
        tasks = tasks[:limit]

    results_log = []
    correct = 0
    total = 0

    for task in tqdm(tasks):
        task_id = task['id']
        # Extract train and test from task['data']
        task_data = task['data']
        # We need to wrap it to match format_arc_prompt's expectation
        # format_arc_prompt expects example['train'] and example['test']
        example = task_data 
        prompt, target_grid = format_arc_prompt(example)
        target_str = format_grid(target_grid)

        if use_mfr:
            # Phase 1: Model Construction
            phase1_prompt = construct_phase1_prompt(prompt)
            output1 = llm(phase1_prompt, max_tokens=1024, stop=["###"])
            model_output = output1['choices'][0]['text']
            
            # Phase 2: Reasoning
            phase2_prompt = construct_phase2_prompt(prompt, model_output)
            output2 = llm(phase2_prompt, max_tokens=2048, stop=["###"])
            generated = output2['choices'][0]['text']
        else:
            output = llm(prompt, max_tokens=2048, stop=["###"])
            generated = output['choices'][0]['text']

        generated_stripped = generated.strip()
        gen_clean = generated_stripped.split("\n")[0].replace(" ", "") if generated_stripped else ""
        
        is_match = (gen_clean == target_str.replace(" ", ""))
        if not is_match:
            # Fallback search
             is_match = target_str.replace(" ", "") in generated.replace(" ", "")[-len(target_str)*3:]

        if is_match:
            correct += 1
        
        total += 1
        results_log.append({
            "task_id": task_id,
            "correct": is_match,
            "generated": generated,
            "target": target_str
        })

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

    with open("arc_gguf_results.json", "w") as f:
        json.dump({"summary": {"accuracy": accuracy, "total": total}, "details": results_log}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mfr", action="store_true")
    args = parser.parse_args()
    evaluate_arc_gguf(args.model, limit=args.limit, use_mfr=args.mfr)
