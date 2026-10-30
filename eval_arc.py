"""Explicit opt-in Unsloth ARC evaluation using the shared exact subset scorer."""

import argparse
import os
import sys

from arc_eval import (
    calculate_confidence_interval, evaluate_tasks, extract_grid, format_arc_prompt,
    format_grid, load_tasks, save_report,
)


def evaluate_arc(model_id, checkpoint_path, split="validation", limit=None, use_mfr=False,
                 data_path="ARC-AGI-master/data/evaluation", output_path="arc_evaluation_results.json"):
    if split != "validation":
        raise ValueError("Only the explicitly supplied local evaluation directory is supported")
    tasks = load_tasks(data_path, limit=limit)
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    os.environ["DISABLE_TORCHAO"] = "1"
    sys.modules["torchao"] = None
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    from unsloth import FastLanguageModel
    import torch

    torch.manual_seed(42)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path or model_id, max_seq_length=6144,
        load_in_4bit=True, device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(prompt, max_tokens):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if inputs.input_ids.shape[1] + max_tokens > 6144:
            raise ValueError("Prompt plus generation budget exceeds the declared context")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    report = evaluate_tasks(tasks, generate, use_mfr=use_mfr)
    save_report(report, output_path, {
        "model": checkpoint_path or model_id, "backend": "unsloth", "load_in_4bit": True,
        "seed": 42, "do_sample": False, "num_shots": 1, "mfr": use_mfr,
        "runtime_geometric_adapter_verified": False,
        "scope": "Loaded model only; no claim that separate IGBundle weights were restored",
    })
    print(f"Exact task solves: {report['summary']['correct']}/{report['summary']['total']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="ARC-AGI-master/data/evaluation")
    parser.add_argument("--output", default="arc_evaluation_results.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mfr", action="store_true")
    args = parser.parse_args()
    evaluate_arc("Qwen/Qwen2.5-7B", args.checkpoint, limit=args.limit, use_mfr=args.mfr,
                 data_path=args.data_path, output_path=args.output)
