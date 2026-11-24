"""Compare task-specific residual logits on a frozen local Qwen2 checkpoint."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
from torch.nn import functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from igbundle.modules.delta_fiber import DeltaFiberUpdate
from scripts.compare_delta_recall import (
    BATCH_SIZE, LEARNING_RATE, SEEDS, event_ids, first_write_labels, make_batch,
    query_logits, swap_write_rounds,
)


SYMBOLS = [*(" " + letter for letter in "ABCDEFGHIJKL"), "0", "1"]
DECAYS = (0.5, 0.8)


def token_ids(tokenizer):
    encoded = [tokenizer.encode(symbol, add_special_tokens=False) for symbol in SYMBOLS]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(SYMBOLS):
        raise ValueError("The task requires 14 distinct single-token symbols in this tokenizer")
    return torch.tensor([ids[0] for ids in encoded])


@torch.no_grad()
def frozen_outputs(model, vocabulary, inputs):
    ids = vocabulary[event_ids(inputs)]
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    logits = model.lm_head(hidden[:, -1]).float()
    answers = logits.index_select(-1, vocabulary[-2:])
    logits[:, vocabulary[-2:]] = -torch.inf
    grouped = torch.cat((answers, logits.logsumexp(dim=-1, keepdim=True)), dim=-1)
    return hidden.float().unsqueeze(2), grouped, logits.amax(dim=-1)


def residual_logits(adapter, features, base_logits, reset):
    correction = query_logits(adapter, features, reset)
    return torch.cat((base_logits[:, :2] + correction, base_logits[:, 2:]), dim=-1)


def fit(arms, model, vocabulary, seed, steps, excluded):
    generator = torch.Generator().manual_seed(seed)
    optimizers = {name: torch.optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)
                  for name, (adapter, _) in arms.items()}
    rejected = 0
    for _ in range(steps):
        batch, labels = [], []
        while sum(len(chunk) for chunk in batch) < BATCH_SIZE:
            size = BATCH_SIZE - sum(len(chunk) for chunk in batch)
            inputs, targets = make_batch(generator, size, rounds=2)
            held_out = torch.tensor(
                [tuple(row) in excluded for row in event_ids(inputs).tolist()], dtype=torch.bool
            )
            rejected += int(held_out.sum())
            batch.append(inputs[~held_out])
            labels.append(targets[~held_out])
        features, base_logits, _ = frozen_outputs(model, vocabulary, torch.cat(batch))
        targets = torch.cat(labels)
        for name, (adapter, reset) in arms.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(residual_logits(adapter, features, base_logits, reset), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite next-token loss in {name}")
            loss.backward()
            optimizer.step()
    return rejected


def score(original, swapped, other_max, labels, first_labels):
    answer_logits = original[:, :2]
    correct = answer_logits.argmax(dim=-1) == labels
    predicted_value = answer_logits.gather(1, labels[:, None]).squeeze(1)
    changed = labels != first_labels
    paired = correct & (swapped[:, :2].argmax(dim=-1) == first_labels)
    return {
        "answer_choice_correct": int(correct.sum()),
        "full_vocab_strict_top1_correct": int((correct & (predicted_value > other_max)).sum()),
        "full_vocab_mean_nll": float(F.cross_entropy(original, labels)),
        "answer_choice_mean_nll": float(F.cross_entropy(answer_logits, labels)),
        "changed_pairs_both_correct": int(paired[changed].sum()),
    }


def run(checkpoint, steps, eval_size, threads):
    torch.set_num_threads(threads)
    weight_file = checkpoint / "model.safetensors"
    if not weight_file.is_file():
        raise ValueError("Expected a local Qwen2 checkpoint with one model.safetensors file")
    with weight_file.open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    vocabulary = token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True,
    ).eval().requires_grad_(False)
    if not isinstance(model, Qwen2ForCausalLM):
        raise ValueError("Expected a Qwen2ForCausalLM checkpoint")

    report = {
        "weight_sha256": weight_hash,
        "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "transformers": transformers.__version__, "device": "cpu", "cpu_threads": threads,
        "backbone_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "token_ids": dict(zip(SYMBOLS, vocabulary.tolist())),
        "steps": steps, "batch_size": BATCH_SIZE, "eval_size_per_seed": eval_size,
        "learning_rate": LEARNING_RATE, "runs": [],
    }
    for seed in SEEDS:
        print(f"Training residual arms for seed {seed}", file=sys.stderr, flush=True)
        torch.manual_seed(seed)
        delta = DeltaFiberUpdate(model.config.hidden_size - 2, 2, mem_dim=16, num_heads=2)
        with torch.no_grad():
            # Every arm initially reproduces the frozen next-token logits.
            delta.out_proj.weight.zero_()
        arms = {"delta": (delta, False), "reset": (deepcopy(delta), True)}
        for decay in (1.0, *DECAYS):
            additive = deepcopy(delta)
            additive.erase = False
            additive.decay = decay
            name = "additive" if decay == 1.0 else f"decaying_additive_{decay}"
            arms[name] = additive, False

        # Exclude both orders of every evaluation prefix from training.
        inputs, labels = make_batch(torch.Generator().manual_seed(seed + 10_000), eval_size, 2)
        swapped = swap_write_rounds(inputs)
        excluded = {tuple(row) for row in event_ids(torch.cat((inputs, swapped))).tolist()}
        rejected = fit(arms, model, vocabulary, seed + 1, steps, excluded)
        original_batched, swapped_batched = (
            [frozen_outputs(model, vocabulary, part) for part in order.split(BATCH_SIZE)]
            for order in (inputs, swapped)
        )
        features, base, other_max = map(torch.cat, zip(*original_batched))
        swapped_features, swapped_base, _ = map(torch.cat, zip(*swapped_batched))
        first_labels = first_write_labels(inputs)
        result = {
            "seed": seed, "held_out_prefix_count": len(excluded),
            "rejected_training_prefixes": rejected,
            "allocated_parameters_per_arm": sum(p.numel() for p in delta.parameters()),
            "changed_pairs": int((labels != first_labels).sum()),
            "base": score(base, swapped_base, other_max, labels, first_labels),
        }
        with torch.no_grad():
            for name, (adapter, reset) in arms.items():
                result[name] = score(
                    residual_logits(adapter, features, base, reset),
                    residual_logits(adapter, swapped_features, swapped_base, reset),
                    other_max, labels, first_labels,
                )
        report["runs"].append(result)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.eval_size, args.threads) < 1 or args.eval_size > 4096:
        parser.error("steps, eval size and threads must be positive, with eval size at most 4096")
    print(json.dumps(run(args.checkpoint, args.steps, args.eval_size, args.threads),
                     indent=2, allow_nan=False))
