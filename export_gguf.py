"""LoRA-only GGUF export. Nonlinear IGBundle operators are not exported."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.inspect_gguf import inspect


def plan_export(base_model_id, checkpoint_path, output_dir, *, converter=None,
                lora_only=False):
    """Validate local prerequisites without importing or downloading a model."""
    if not lora_only:
        raise ValueError("Explicit --lora-only acknowledgement is required; geometry is not exported")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError("An explicit base model identifier is required")
    checkpoint = Path(checkpoint_path).resolve()
    config_path = checkpoint / "adapter_config.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LORA adapter_config.json")
    if config.get("base_model_name_or_path") != base_model_id:
        raise ValueError("Base identifier must exactly match adapter_config.json")
    if not any((checkpoint / name).is_file()
               for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError("Required PEFT adapter weights are missing")
    if converter is None:
        raise ValueError("An explicit --converter Python script is required")
    converter_path = Path(converter).resolve()
    if not converter_path.is_file() or converter_path.suffix.lower() != ".py":
        raise FileNotFoundError(f"Converter Python script is missing: {converter_path}")

    prefix = Path(output_dir).resolve()
    gguf = Path(f"{prefix}.gguf")
    merged = Path(f"{prefix}_hf")
    for path in (gguf, merged):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    return {
        "base_model": base_model_id,
        "checkpoint": str(checkpoint),
        "merged_hf": str(merged),
        "gguf": str(gguf),
        "converter": str(converter_path),
        "scope": "LoRA merge only; no runtime IGBundle or hypercomplex operators",
        "runtime_geometric_adapter_exported": False,
    }


def export_to_gguf_ready(base_model_id, checkpoint_path, output_dir, *,
                         converter=None, lora_only=False, dry_run=False):
    plan = plan_export(base_model_id, checkpoint_path, output_dir,
                       converter=converter, lora_only=lora_only)
    if dry_run:
        return plan

    os.environ["DISABLE_TORCHAO"] = "1"
    sys.modules["torchao"] = None
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Reserve fresh destinations before heavyweight allocation; failures remain visible.
    merged = Path(plan["merged_hf"])
    merged.mkdir(parents=True, exist_ok=False)
    with Path(plan["gguf"]).open("xb"):
        pass
    print(plan["scope"])
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="cpu", torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, plan["checkpoint"], device_map="cpu")
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(plan["merged_hf"])
    tokenizer.save_pretrained(plan["merged_hf"])
    subprocess.run(
        [sys.executable, plan["converter"], plan["merged_hf"],
         "--outfile", plan["gguf"], "--outtype", "f16"],
        cwd=str(Path(plan["converter"]).parent), check=True,
    )
    directory = inspect(plan["gguf"])
    if directory["tensor_count"] == 0:
        raise ValueError("Converter output has no tensors")
    plan["directory_inspection"] = directory
    print(f"LoRA-only export produced: {plan['gguf']} (runtime execution unverified)")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New output prefix, without extension")
    parser.add_argument("--converter", required=True, help="Trusted Python HF-to-GGUF converter")
    parser.add_argument("--lora-only", action="store_true",
                        help="Acknowledge omission of nonlinear geometric operators")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check local paths/config only; no model imports or allocation")
    args = parser.parse_args()
    plan = export_to_gguf_ready(
        args.base, args.checkpoint, args.output, converter=args.converter,
        lora_only=args.lora_only, dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
