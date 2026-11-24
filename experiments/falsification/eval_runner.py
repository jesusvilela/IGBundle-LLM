import argparse
import json
import os
import subprocess
import time
from datetime import datetime
import numpy as np

def run_lm_eval_task(model_path, task, limit=200, n_runs=10):
    print(f"=== Running Tier 1 Benchmark: {task} ===")
    results = []
    
    # We will simulate the multiple runs to get mean/std, or run it 10 times with different seeds.
    # Note: lm_eval doesn't natively do "10 runs with mean/std" unless we loop it.
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs} for {task}...")
        cmd = [
            r"unsloth_env\Scripts\python.exe", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True,load_in_4bit=True",
            "--tasks", task,
            "--limit", str(limit),
            "--device", "cuda:0",
            "--batch_size", "4",
            "--seed", str(42 + i)
        ]
        
        try:
            # We redirect output, but capture it for parsing
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            # Find the accuracy in the stdout. 
            # We'll need a better parser depending on lm_eval output format, but let's mock the extraction or parse the JSON.
            # Ideally we use --output_path and parse the JSON.
            output_dir = f"lm_eval_tmp_{task}_{i}"
            cmd.extend(["--output_path", output_dir])
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            # Find the generated json
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    if f.endswith(".json"):
                        with open(os.path.join(output_dir, f), "r") as jf:
                            data = json.load(jf)
                            # the key depends on the task
                            # e.g., data["results"]["gsm8k"]["exact_match,strict-match"]
                            res_dict = data.get("results", {}).get(task, {})
                            acc = res_dict.get("acc,none", res_dict.get("exact_match,strict-match", 0.0))
                            results.append(acc)
        except Exception as e:
            print(f"Error running {task}: {e}")
            results.append(0.0)
            
    # Calculate stats
    if results:
        mean_acc = float(np.mean(results))
        std_acc = float(np.std(results))
    else:
        mean_acc = 0.0
        std_acc = 0.0
        
    return {
        "mean": mean_acc,
        "std": std_acc,
        "n_runs": n_runs,
        "n_problems": limit
    }

def run_tier2_arc(checkpoint_path):
    print(f"=== Running Tier 2: ARC-AGI Validations ===")
    cmd = [
        r"unsloth_env\Scripts\python.exe", "experiments/falsification/eval_runner_arc.py",
        "--checkpoint", checkpoint_path if checkpoint_path != "Qwen/Qwen2.5-7B" else "",
        "--limit", "120",
        "--output", "arc_tmp_results.json"
    ]
    if checkpoint_path == "Qwen/Qwen2.5-7B":
        cmd.remove("--checkpoint")
        cmd.remove("")
    
    subprocess.run(cmd, check=False)
    
    if os.path.exists("arc_tmp_results.json"):
        with open("arc_tmp_results.json", "r") as f:
            return json.load(f)
    return {}

def run_tier3_faithfulness_audit(checkpoint_path):
    print(f"=== Running Tier 3: Manifold Faithfulness Audit ===")
    # We would load the GeometricIGBundleAdapter and run random batches to assert K, S, JSD, etc.
    # For now, placeholder indicating we will run the geometry validation.
    return {
        "curvature_stability": {"passed": False, "variance": 0.0},
        "fiber_section_entropy": {"passed": False, "value": 0.0},
        "fiber_diversity_jsd": {"passed": False, "value": 0.0},
        "parallel_transport_fidelity": {"passed": False, "ratio": 1.0},
        "geodesic_triangle_inequality": {"passed": False, "satisfaction_rate": 0.0},
        "generic_energy_conservation": {"passed": False, "drift": 1.0}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="A|B|C|D")
    parser.add_argument("--checkpoint", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output", default="experiments/falsification/results/eval_results.json")
    parser.add_argument("--skip_tier1", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    output_data = {
        "config": args.config,
        "checkpoint_path": args.checkpoint,
        "eval_timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmarks": {},
        "tier2_arc": {},
        "tier3_faithfulness": {},
        "trainable_params": 0, # Will be filled from checkpoint config
        "training_steps": 1000,
    }

    if not args.skip_tier1:
        output_data["benchmarks"]["gsm8k"] = run_lm_eval_task(args.checkpoint, "gsm8k", limit=200, n_runs=10)
        output_data["benchmarks"]["mmlu"] = run_lm_eval_task(args.checkpoint, "mmlu", limit=200, n_runs=10)
        output_data["benchmarks"]["arc_easy"] = run_lm_eval_task(args.checkpoint, "arc_easy", limit=200, n_runs=10)

    # Run Tier 2
    output_data["tier2_arc"] = run_tier2_arc(args.checkpoint)
    
    # Run Tier 3 (Only if it's Config A geometric)
    if args.config == "A":
        output_data["tier3_faithfulness"] = run_tier3_faithfulness_audit(args.checkpoint)

    output_data["wall_time_seconds"] = int(time.time() - start_time)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
