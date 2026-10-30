"""Explicit opt-in GGUF ARC evaluation using the shared exact subset scorer."""

import argparse
from pathlib import Path

from arc_eval import evaluate_tasks, format_arc_prompt, format_grid, load_tasks, save_report


def evaluate_arc_gguf(model_path, data_path="ARC-AGI-master/data/evaluation", limit=None,
                      use_mfr=False, output_path="arc_gguf_results.json",
                      n_gpu_layers=0, n_threads=1):
    tasks = load_tasks(data_path, limit=limit)
    if Path(output_path).exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    from llama_cpp import Llama

    llm = Llama(model_path=model_path, n_ctx=6144, n_gpu_layers=n_gpu_layers,
                n_threads=n_threads, seed=42)

    def generate(prompt, max_tokens):
        if len(llm.tokenize(prompt.encode("utf-8"))) + max_tokens > 6144:
            raise ValueError("Prompt plus generation budget exceeds the declared context")
        return llm(prompt, max_tokens=max_tokens, temperature=0.0,
                   seed=42, stop=["###"])["choices"][0]["text"]

    report = evaluate_tasks(tasks, generate, use_mfr=use_mfr)
    save_report(report, output_path, {
        "model": model_path, "backend": "llama_cpp", "seed": 42, "temperature": 0.0,
        "num_shots": 1, "mfr": use_mfr, "n_gpu_layers": n_gpu_layers, "n_threads": n_threads,
        "runtime_geometric_adapter_verified": False,
        "scope": "GGUF model only; no runtime IGBundle module is injected by this runner",
    })
    print(f"Exact task solves: {report['summary']['correct']}/{report['summary']['total']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", default="ARC-AGI-master/data/evaluation")
    parser.add_argument("--output", default="arc_gguf_results.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mfr", action="store_true")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    evaluate_arc_gguf(args.model, args.data_path, args.limit, args.mfr, args.output,
                      args.gpu_layers, args.threads)
