"""Train and compare paired delta-rule recall models on CPU."""

import argparse
import copy
import json
import platform

import torch
from torch.nn import functional as F

from igbundle.modules.delta_fiber import DeltaFiberUpdate


KEYS = 4
SEEDS = (11, 29, 47)
STEPS = 200
BATCH_SIZE = 32
EVAL_SIZE = 512
LEARNING_RATE = 0.01


def make_batch(generator, size, rounds=1):
    """Write independent bits for every key in each round, then query the last round."""
    orders, values = [], []
    for _ in range(rounds):
        orders.append(torch.rand(size, KEYS, generator=generator).argsort(dim=-1))
        values.append(torch.randint(2, (size, KEYS), generator=generator))
    query_index = torch.randint(KEYS, (size, 1), generator=generator)
    query = orders[-1].gather(1, query_index)
    labels = values[-1].gather(1, query_index).squeeze(1)
    order = torch.cat(orders, dim=1)
    bits = torch.cat(values, dim=1)
    key_features = F.one_hot(torch.cat((order, query), dim=1), num_classes=KEYS)
    # The query carries zero value input, independent of the correct bit.
    value_features = F.pad(F.one_hot(bits, num_classes=2), (0, 0, 0, 1))
    inputs = torch.cat((key_features, value_features), dim=-1)
    return inputs.to(torch.float32).unsqueeze(2), labels


def query_logits(model, inputs, reset):
    # In the reset arm every token gets its own zero memory. With supervision
    # only at the query, this is equivalent to evaluating just that token.
    sequence = inputs[:, -1:] if reset else inputs
    return model(sequence)[:, -1, 0]


def fit(model, seed, reset, rounds):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(STEPS):
        inputs, labels = make_batch(generator, BATCH_SIZE, rounds)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(query_logits(model, inputs, reset), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite recall training loss")
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, inputs, labels, reset, changed):
    logits = query_logits(model.eval(), inputs, reset)
    correct = logits.argmax(dim=-1) == labels
    metrics = {
        "correct": int(correct.sum()),
        "mean_nll": float(F.cross_entropy(logits, labels)),
    }
    if changed is not None:
        metrics["correct_when_value_changed"] = int(correct[changed].sum())
    return metrics


@torch.no_grad()
def evaluate_round_order(model, inputs, labels, first_labels, changed):
    """Count pairs where the model follows the latest label in both orders."""
    swapped = torch.cat((inputs[:, KEYS:2 * KEYS], inputs[:, :KEYS], inputs[:, -1:]), dim=1)
    original_prediction = query_logits(model.eval(), inputs, False).argmax(dim=-1)
    swapped_prediction = query_logits(model, swapped, False).argmax(dim=-1)
    both_correct = (original_prediction == labels) & (swapped_prediction == first_labels)
    return int(both_correct[changed].sum())


def run(rounds):
    torch.set_num_threads(1)
    report = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "keys": KEYS,
        "write_rounds": rounds,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "eval_size_per_seed": EVAL_SIZE,
        "learning_rate": LEARNING_RATE,
        "runs": [],
    }
    for seed in SEEDS:
        torch.manual_seed(seed)
        recurrent = DeltaFiberUpdate(KEYS, 2, mem_dim=16, num_heads=2)
        reset = copy.deepcopy(recurrent)
        additive = copy.deepcopy(recurrent)
        additive.erase = False
        fit(recurrent, seed + 1, reset=False, rounds=rounds)
        fit(reset, seed + 1, reset=True, rounds=rounds)
        fit(additive, seed + 1, reset=False, rounds=rounds)
        inputs, labels = make_batch(
            torch.Generator().manual_seed(seed + 10_000), EVAL_SIZE, rounds
        )
        changed = None
        if rounds == 2:
            first = inputs[:, :KEYS, 0]
            query_key = inputs[:, -1:, 0, :KEYS]
            matched = (first[:, :, :KEYS] * query_key).sum(dim=-1)
            first_labels = (first[:, :, KEYS + 1] * matched).sum(dim=-1).long()
            changed = first_labels != labels

        result = {
            "seed": seed,
            "allocated_parameters_per_arm": sum(p.numel() for p in recurrent.parameters()),
            "majority_class_correct": max(int(labels.sum()), EVAL_SIZE - int(labels.sum())),
            "recurrent": evaluate(recurrent, inputs, labels, False, changed),
            "additive": evaluate(additive, inputs, labels, False, changed),
            "reset": evaluate(reset, inputs, labels, True, changed),
            "recurrent_weights_with_reset": evaluate(recurrent, inputs, labels, True, changed),
        }
        if changed is not None:
            result["queries_with_changed_value"] = int(changed.sum())
            for name, model in (("recurrent", recurrent), ("additive", additive)):
                result[name]["both_round_orders_correct_when_changed"] = evaluate_round_order(
                    model, inputs, labels, first_labels, changed
                )
        report["runs"].append(result)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-rounds", type=int, choices=(1, 2), default=1,
        help="Use two rounds to test recall after overwriting every key.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.write_rounds), indent=2, allow_nan=False))
