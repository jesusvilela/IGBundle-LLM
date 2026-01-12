import argparse
import json
import os
import sys
import logging
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def map_user_benchmark_to_task(benchmark_name, task_manager):
    """
    Maps user-friendly benchmark names to lm_eval task names.
    """
    available_tasks = task_manager.all_tasks
    
    # Direct mappings or fuzzy search
    mappings = {
        "mmlu-pro": ["mmlu_pro", "mmlu"],
        "mmlu": ["mmlu"],
        "aime25": ["aime", "math_algebra", "gsm8k"], # AIME might not be present, fallback to math
        "gpqa": ["gpqa"],
        "livecodebench": ["livecodebench", "humaneval"], # Fallback
        "minif2f": ["minif2f", "math"],
        "scicode": ["scicode"],
        "arc": ["arc_challenge", "arc_easy"],
        "truthfulqa": ["truthfulqa"]
    }

    candidates = mappings.get(benchmark_name.lower(), [])
    for cand in candidates:
        if cand in available_tasks:
            return cand
    
    # If no direct match, check if it's a substring
    for task in available_tasks:
        if benchmark_name.lower() in task:
            return task
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on GGUF model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["mmlu", "gpqa"], help="List of benchmarks to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task")
    parser.add_argument("--batch_size", type=str, default="auto")
    args = parser.parse_args()

    print(f"Initializing Benchmark Runner for: {args.model_path}")
    
    tm = TaskManager()
    print("Loading available tasks...")
    # This might take a moment
    all_tasks = tm.all_tasks
    
    selected_tasks = []
    for b in args.benchmarks:
        mapped = map_user_benchmark_to_task(b, tm)
        if mapped:
            print(f"Mapped user benchmark '{b}' -> '{mapped}'")
            selected_tasks.append(mapped)
        else:
            print(f"Warning: Could not find task for '{b}'. Skipping.")

    if not selected_tasks:
        print("No valid tasks selected. Exiting.")
        return

    print(f"Running evaluation on: {selected_tasks}")

    # Construct model args for GGUF
    # lm_eval uses 'gguf' or 'llama_cpp' model type
    model_args = f"pretrained={args.model_path},n_gpu_layers=-1" 

    try:
        results = simple_evaluate(
            model="gguf",
            model_args=model_args,
            tasks=selected_tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            task_manager=tm
        )
        
        print("\n=== BENCHMARK RESULTS ===")
        print(json.dumps(results["results"], indent=2))
        
        output_file = "benchmark_results_comprehensive.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
