"""
Falsification Experiment: Geometry vs. Layer Count
Unified training script for configs A, B, C.

Config A: Single layer 12, rank 8, WITH geometric adapter + losses
Config B: Layers 8-20, rank 2, vanilla LoRA (parameter-matched to A)
Config C: Layers 8-20, rank 8, vanilla LoRA (full parameter budget)

All configs: same data, same base model, same seed, same compute steps.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("falsification.train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Configuration ─────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    config_name: str            # A, B, or C
    lora_layers: list           # Which transformer layers get LoRA
    lora_rank: int              # LoRA rank
    lora_alpha: int             # LoRA alpha (2 * rank)
    use_geometric: bool         # Whether to use IGBundle geometric adapter
    trainable_params_approx: int

CONFIGS = {
    "A": ExperimentConfig(
        config_name="A",
        lora_layers=[12],       # Not used directly — geometric adapter hooks layer 12
        lora_rank=8,
        lora_alpha=16,
        use_geometric=True,
        trainable_params_approx=524_000,
    ),
    "B": ExperimentConfig(
        config_name="B",
        lora_layers=list(range(8, 21)),  # layers 8-20
        lora_rank=2,
        lora_alpha=4,
        use_geometric=False,
        trainable_params_approx=507_000,
    ),
    "C": ExperimentConfig(
        config_name="C",
        lora_layers=list(range(8, 21)),  # layers 8-20
        lora_rank=8,
        lora_alpha=16,
        use_geometric=False,
        trainable_params_approx=2_600_000,
    ),
}

# Fixed hyperparameters (same for all configs per spec)
# NOTE: batch_size reduced from 4→1 and grad_accum 4→16 to fit RTX 3060 Ti 8GB
# Effective batch size remains 16 (1 * 16 = 16, same as original 4 * 4 = 16)
MAX_STEPS = 500
BASE_LR = 2e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRAD_ACCUM = 8
BATCH_SIZE = 1
MAX_SEQ_LEN = 128
SEED = 42
CHECKPOINT_EVERY = 100


# ── Dataset ───────────────────────────────────────────────────────

class TextQADataset(IterableDataset):
    """Streaming dataset from full_scale_train.jsonl with ChatML template.

    Uses line-offset index to avoid loading entire file into memory.
    Reads and shuffles offsets (ints) instead of full string lines.
    """

    def __init__(self, path: str, tokenizer, max_len: int, seed: int = 42):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        # Build line-offset index (only stores byte offsets, ~8 bytes each)
        self._offsets = []
        with open(self.path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(pos)
        logger.info(f"Dataset indexed: {len(self._offsets):,} lines from {path}")

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            indices = list(range(len(self._offsets)))
            rng.shuffle(indices)
            with open(self.path, "r", encoding="utf-8") as f:
                for idx in indices:
                    f.seek(self._offsets[idx])
                    line = f.readline().strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    q = item.get("input", "")
                    a = item.get("output", "")
                    if not q or not a:
                        continue

                    text = (
                        f"<|im_start|>user\n{q}<|im_end|>\n"
                        f"<|im_start|>assistant\n{a}<|im_end|>"
                    )
                    enc = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_len,
                        padding="max_length",
                    )
                    yield {
                        "input_ids": enc.input_ids.squeeze(0),
                        "attention_mask": enc.attention_mask.squeeze(0),
                        "labels": enc.input_ids.squeeze(0),
                    }


# ── Vanilla LoRA Training (Configs B, C) ──────────────────────────

def train_vanilla_lora(args, exp_cfg: ExperimentConfig):
    """Standard PEFT LoRA training — no geometric losses."""
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info(f"=== Config {exp_cfg.config_name}: Vanilla LoRA ===")
    logger.info(f"Layers: {exp_cfg.lora_layers}, Rank: {exp_cfg.lora_rank}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "30GiB"},
        trust_remote_code=True,
    )

    # Apply LoRA to specific layers
    # PEFT's target_modules applies to ALL layers by default.
    # To restrict to specific layers, we use layers_to_transform.
    lora_config = LoraConfig(
        r=exp_cfg.lora_rank,
        lora_alpha=exp_cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=exp_cfg.lora_layers,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # Dataset
    dataset = TextQADataset(args.data_path, tokenizer, MAX_SEQ_LEN, SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Optimizer + scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - WARMUP_STEPS)

    # Output dir
    output_dir = Path(args.output_dir) / f"config_{exp_cfg.config_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()
    log_interval = 10

    data_iter = iter(loader)
    while step < MAX_STEPS:
        # Accumulate gradients
        for _ in range(GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        # LR warmup
        if step < WARMUP_STEPS:
            lr_scale = (step + 1) / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg["lr"] = BASE_LR * lr_scale
        else:
            scheduler.step()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        should_log = (step % log_interval == 0) or (step <= 5)
        if should_log:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            n = min(step, log_interval) or 1
            logger.info(
                f"[{exp_cfg.config_name}] step={step}/{MAX_STEPS} "
                f"loss={accum_loss/n:.4f} lr={lr:.2e} "
                f"elapsed={elapsed:.0f}s"
            )
            accum_loss = 0.0

        if step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            logger.info(f"Saved checkpoint: {ckpt_dir}")

        # VRAM cleanup
        if step % 64 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall_time = time.time() - start_time
    logger.info(f"Config {exp_cfg.config_name} training complete. {wall_time:.0f}s")

    # Save training metadata
    meta = {
        "config": exp_cfg.config_name,
        "trainable_params": trainable,
        "total_params": total,
        "max_steps": MAX_STEPS,
        "final_checkpoint": str(output_dir / f"checkpoint-{MAX_STEPS}"),
        "wall_time_seconds": int(wall_time),
        "lora_rank": exp_cfg.lora_rank,
        "lora_layers": exp_cfg.lora_layers,
        "learning_rate": BASE_LR,
        "seed": SEED,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ── Geometric LoRA Training (Config A) ────────────────────────────

def train_geometric(args, exp_cfg: ExperimentConfig):
    """Config A: Single-layer geometric adapter with full IGBundle losses.

    This reuses the existing geometric adapter infrastructure but with
    controlled hyperparameters matching the experiment spec.
    """
    logger.info(f"=== Config A: Geometric IGBundle Adapter ===")

    # Add project root to path for imports
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.igbundle.modules.geometric_adapter import create_geometric_adapter
    from src.igbundle.core.config import IGBundleConfig

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "30GiB"},
        trust_remote_code=True,
    )
    model.requires_grad_(False)
    # Create geometric adapter (same config as original training)
    config = IGBundleConfig(hidden_size=3584, use_dynamics=True)
    adapter = create_geometric_adapter(config).to(DEVICE)
    adapter.train()

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    logger.info(f"Geometric adapter trainable params: {trainable:,}")

    # Hook adapter into layer 12
    layers = None
    for accessor in [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.base_model.model.model.layers,
    ]:
        try:
            layers = accessor(model)
            break
        except (AttributeError, IndexError):
            continue
    if layers is None:
        raise RuntimeError("Cannot find transformer layers in model")

    hook_handle = None
    adapter_output = {}

    def adapter_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.enable_grad():
            adapted, geo_state = adapter(hidden.float())
            # Perturbation clamping: delta_norm <= 0.1 * base_norm
            delta = adapted - hidden.float()
            base_norm = hidden.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            delta_norm = delta.norm(dim=-1, keepdim=True)
            scale = torch.clamp(0.1 * base_norm / delta_norm.clamp(min=1e-6), max=1.0)
            clamped = hidden.float() + delta * scale
            adapter_output["geo_state"] = geo_state
        if isinstance(output, tuple):
            return (clamped.half(),) + output[1:]
        return clamped.half()

    hook_handle = layers[12].register_forward_hook(adapter_hook)

    # Dataset — same batch_size and grad_accum as vanilla configs
    dataset = TextQADataset(args.data_path, tokenizer, MAX_SEQ_LEN, SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    # Optimizer: adapter params only
    optimizer = AdamW(
        adapter.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - WARMUP_STEPS)

    # Output dir
    output_dir = Path(args.output_dir) / "config_A"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Geo loss ramp (same as alignment stage)
    geo_lambda_max = 0.05
    geo_ramp_steps = 300

    # Training loop
    step = 0
    accum_loss = 0.0
    accum_geo_loss = 0.0
    optimizer.zero_grad()
    start_time = time.time()
    log_interval = 10

    logger.info(f"Starting training loop (max_steps={MAX_STEPS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM})")
    data_iter = iter(loader)
    while step < MAX_STEPS:
        for i in range(GRAD_ACCUM):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            adapter_output.clear()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            llm_loss = outputs.loss

            # Geometric losses from adapter
            geo_loss = torch.tensor(0.0, device=DEVICE)
            geo_state = adapter_output.get("geo_state")
            if geo_state is not None:
                geo_losses = adapter.compute_geometric_losses(geo_state)
                for name, val in geo_losses.items():
                    if torch.is_tensor(val) and val.requires_grad:
                        geo_loss = geo_loss + val

            # Homotopy ramp
            ramp = min(1.0, step / max(geo_ramp_steps, 1))
            total_loss = (llm_loss + ramp * geo_lambda_max * geo_loss) / GRAD_ACCUM
            total_loss.backward()
            accum_loss += llm_loss.item() / GRAD_ACCUM
            accum_geo_loss += geo_loss.item() / GRAD_ACCUM

        # torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if (step % log_interval == 0) or (step <= 5):
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            # Compute S from geo_state if available
            s_val = "N/A"
            gs = adapter_output.get("geo_state")
            if gs is not None and gs.fiber_sections is not None:
                with torch.no_grad():
                    p = gs.fiber_sections.clamp(min=1e-8)
                    s_val = f"{float(-(p * p.log()).sum(-1).mean()):.3f}"
            n = min(step, log_interval) or 1
            logger.info(
                f"[A] step={step}/{MAX_STEPS} "
                f"llm_loss={accum_loss/n:.4f} "
                f"geo_loss={accum_geo_loss/n:.4f} "
                f"S={s_val} lr={lr:.2e} elapsed={elapsed:.0f}s"
            )
            accum_loss = 0.0
            accum_geo_loss = 0.0

        if step % CHECKPOINT_EVERY == 0 or step == MAX_STEPS:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(adapter.state_dict(), ckpt_dir / "adapter.pt")
            tokenizer.save_pretrained(str(ckpt_dir))
            # Also save the full model so eval_runner can load it uniformly
            model.save_pretrained(str(ckpt_dir))
            logger.info(f"Saved checkpoint: {ckpt_dir}")

        if step % 64 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    hook_handle.remove()
    wall_time = time.time() - start_time
    logger.info(f"Config A training complete. {wall_time:.0f}s")

    meta = {
        "config": "A",
        "trainable_params": trainable,
        "max_steps": MAX_STEPS,
        "final_checkpoint": str(output_dir / f"checkpoint-{MAX_STEPS}"),
        "wall_time_seconds": int(wall_time),
        "geo_lambda_max": geo_lambda_max,
        "geo_ramp_steps": geo_ramp_steps,
        "learning_rate": BASE_LR,
        "seed": SEED,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ── Entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Falsification Experiment: Geometry vs. Layer Count"
    )
    parser.add_argument(
        "--config", required=True, choices=["A", "B", "C", "all"],
        help="Which config to train (A/B/C or 'all' for sequential)"
    )
    parser.add_argument(
        "--model-id",
        default="h:/LLM-MANIFOLD/igbundle_qwen7b_cp600",
        help="Base model path (Qwen2.5-7B)"
    )
    parser.add_argument(
        "--data-path",
        default="data/full_scale_train.jsonl",
        help="Training data (JSONL with input/output keys)"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/falsification/checkpoints",
        help="Output directory for checkpoints"
    )
    args = parser.parse_args()

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]

    results = {}
    for cfg_name in configs_to_run:
        exp_cfg = CONFIGS[cfg_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Config {cfg_name}")
        logger.info(f"{'='*60}")

        if exp_cfg.use_geometric:
            meta = train_geometric(args, exp_cfg)
        else:
            meta = train_vanilla_lora(args, exp_cfg)

        results[cfg_name] = meta
        logger.info(f"Config {cfg_name} done: {meta['trainable_params']:,} params, "
                     f"{meta['wall_time_seconds']}s")

        # Free VRAM between configs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Save combined results
    out_path = Path(args.output_dir) / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll training complete. Summary: {out_path}")


if __name__ == "__main__":
    main()
