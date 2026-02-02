import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_results(path_pattern):
    results = {}
    files = glob.glob(path_pattern)
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            if 'results' in data:
                for task, metrics in data['results'].items():
                    # Prioritize acc_norm, then acc, then mc2
                    score = metrics.get('acc_norm,none') or metrics.get('acc,none') or metrics.get('mc2,none') or 0.0
                    results[task] = score * 100 
    return results

def generate_report():
    print("Generating Visual Comparison Report (Optimized Suite)...")
    
    # Validating path (Optimized Suite)
    search_path = "eval_results_opt_merged/**/results_*.json"
    merged_data = load_results(search_path)
    
    if not merged_data:
         # Try concrete path if glob fails
         search_path = "eval_results_opt_merged/output__igbundle_qwen7b_riemannian_merged/results_*.json"
         merged_data = load_results(search_path)

    # Validating path (GSM8K)
    gsm8k_path = "eval_results_gsm8k/**/results_*.json"
    gsm8k_data = load_results(gsm8k_path)
    if not gsm8k_data:
        gsm8k_path = "eval_results_gsm8k/output__igbundle_qwen7b_riemannian_merged/results_*.json"
        gsm8k_data = load_results(gsm8k_path)
    
    merged_data.update(gsm8k_data)

    
    # Baseline Mock Data (We can update this later with real baseline runs if needed)
    # Using 0.0 only where we truly don't know the baseline yet
    baseline_data = {
        "arc_challenge": 54.86,
        "truthfulqa_mc2": 0.0, 
        "winogrande": 0.0,
        "gsm8k": 0.0
    }

    tasks = list(merged_data.keys())
    # Sort for consistent display
    if tasks:
        tasks.sort()

    if not tasks:
        print("No results found yet. Using placeholders.")
        tasks = ["arc_challenge", "gsm8k", "truthfulqa_mc2", "winogrande"]
        merged_scores = [0, 0, 0, 0]
    else:
        merged_scores = [merged_data.get(t, 0) for t in tasks]
        
    baseline_scores = [baseline_data.get(t, 0) for t in tasks]

    # Plotting
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (Qwen2.5)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, merged_scores, width, label='Riemannian (IGBundle)', color='#3498db')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Riemannian Fine-Tuning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()

    output_file = "benchmark_comparison_opt.png"
    plt.savefig(output_file, dpi=300)
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    generate_report()
