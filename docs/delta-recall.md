# Delta-rule associative recall

This CPU experiment ports `DeltaFiberUpdate` from development commit
[`db00915`](https://github.com/jesusvilela/IGBundle-LLM/commit/db00915)
onto the public evidence branch. The port fixes token/component ordering
and applies optional gate features to logits before the sigmoid. These
changes alter the legacy numerics for multiple components or modulated gates.
The default LLM adapter, checkpoint loader and canonical Hamiltonian are
independent of this module.

## Reproduce

From the repository root with Python 3.11, set up an isolated environment with CPU PyTorch.

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
PYTHONPATH=src python -m unittest discover -s tests -p test_delta_fiber.py
PYTHONPATH=src python scripts/compare_delta_recall.py --write-rounds 1
PYTHONPATH=src python scripts/compare_delta_recall.py --write-rounds 2
```

In PowerShell, set `$env:PYTHONPATH = 'src'`, then run the same Python
commands without the `PYTHONPATH=src` prefix. The script prints JSON with
per-seed counts and mean query NLL. The run uses generated tensors and
requires no model weights, datasets or GPU.

Each round writes four randomly ordered keys with independent random
bits. The query supplies a key and no bit, with the last write supplying
the target. All four training arms start with identical weights, draw
the same minibatches and take 200 AdamW steps. Each allocates 386
parameters with one-hot inputs. The additive and decaying additive arms
disable the erase term. The latter retains 80% of the previous memory at
every token. The reset arm passes only the query to the memory module.
Another evaluation resets memory on the trained delta model using the
same weights. Evaluation uses an independent RNG stream from the same
finite synthetic distribution.

In the overwrite test, the same query is evaluated with the two write
rounds in both orders. When its bit changed between rounds, a pair counts
as correct only when the prediction follows the latest write in both
orders. The additive update sums token-local writes, so exchanging the
rounds preserves its prediction in exact arithmetic. The paired test
holds the multiset of input tokens fixed while changing the target.

## Recorded CPU run

Python 3.11.16, PyTorch 2.9.1+cpu on Windows. The script reports correct counts
and mean query negative log likelihood.

**Single write, correct out of 512 per seed**

| Seed | Delta | Additive | Decaying additive | Trained with reset | Delta weights with reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 512 | 512 | 512 | 256 | 262 |
| 29 | 512 | 512 | 512 | 252 | 252 |
| 47 | 512 | 512 | 512 | 253 | 261 |

**Overwrite, correct in original order and in both orders**

| Seed | Delta / 512 | Additive / 512 | Decaying additive / 512 | Trained with reset / 512 | Delta weights with reset / 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 512 | 385 | 509 | 258 | 246 |
| 29 | 512 | 402 | 501 | 231 | 238 |
| 47 | 512 | 404 | 505 | 257 | 254 |

| Seed | Delta on changed pairs | Additive on changed pairs | Decaying additive on changed pairs |
| --- | ---: | ---: | ---: |
| 11 | 254/254 | 0/254 | 247/254 |
| 29 | 245/245 | 0/245 | 224/245 |
| 47 | 231/231 | 0/231 | 211/231 |

The delta model learned to track the most recent write on this four-key
task. It scored 730/730 changed pairs. Fixed decay tracked recency on
682/730 pairs at the default retention of 0.8. An exploratory sweep
on the same three evaluation seeds gave the following counts.

| Retention per token | Original order / 1,536 | Both orders on changed pairs / 730 |
| ---: | ---: | ---: |
| 0.50 | 1,519 | 697 |
| 0.65 | 1,517 | 688 |
| 0.80 | 1,515 | 682 |
| 0.90 | 1,490 | 632 |
| 0.95 | 1,424 | 520 |

Reproduce a row with `python scripts/compare_delta_recall.py --write-rounds 2 --decay 0.5`,
using the same `PYTHONPATH` setup as above. The sweep reuses evaluation data and
leaves performance after selecting a retention factor on independent data open.
Curvature and entropy modulation were inactive in training.

## Frozen tiny Qwen2 features

The optional encoder assigns three token IDs per key, one for each
write value and one for the query. It produces causal hidden states
with a randomly initialized two-layer Qwen2 backbone. The backbone uses
seed 31 and shares the same 4,720 frozen parameters across all arms.
The memory modules use 906 allocated parameters each and train only on
the query classification loss.

```sh
python -m pip install -r requirements-test-hf.txt
PYTHONPATH=src python scripts/compare_delta_recall.py --write-rounds 2 --encoder qwen2
```

Both round orders are encoded afresh. Resetting memory still gives the
classifier the query's contextual backbone state, so the reset baseline
retains prefix information. Global addition can also respond to order
because the contextual vectors change when the order changes.

| Seed | Delta on changed pairs | Additive | Decaying additive | Trained with reset | Delta weights with reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 254/254 | 152/254 | 225/254 | 1/254 | 7/254 |
| 29 | 245/245 | 145/245 | 211/245 | 0/245 | 6/245 |
| 47 | 231/231 | 143/231 | 202/231 | 6/231 | 6/231 |

The delta head retained its overwrite result when the input came from
causal transformer features. The decaying additive head solved 638/730
changed pairs at 0.8 retention and 634/730 at 0.5. The experiment trains a classifier on a synthetic token
vocabulary. It leaves pretrained weights, language-model loss and
autoregressive generation for a separate experiment.

## Pretrained Qwen2.5 residual logits

This experiment maps each synthetic event to one token from a local
pretrained Qwen2 tokenizer. It trains a memory head on the frozen final
hidden states and adds its two corrections to the native logits for
answer tokens `0` and `1`. Every arm starts with a zero correction,
and the base model supplies the same hidden states and logits to all
arms. The unadapted model, additive memory, two decay rates, and the
memory-reset head serve as controls.

```powershell
$env:PYTHONPATH = 'src'
python -m scripts.compare_pretrained_recall 'path-to-local-qwen2-checkpoint' --steps 64 --eval-size 256 --threads 4
```

Use the pinned CPU and Hugging Face dependencies above. The command
requires a local Qwen2 checkpoint with a single `model.safetensors`
file and records the weight and config digests. It trains on the
original orders only and screens out both orders of every evaluation
prefix within the corresponding seed. The target is the token for
the last-written bit. Log-sum-exp over the unchanged vocabulary
entries gives the exact full-vocabulary loss for the two edited
logits. The script reports that loss and full-vocabulary top-1,
alongside two-choice and swapped-order scores.

### Recorded pretrained run

The [JSON report](benchmark-results/pretrained-delta-recall.json) records
three seeds with 256 evaluation queries each after 64 optimizer steps.
The frozen model had 494,032,768 parameters, with 46,666 allocated to
each trainable head. One sampled training prefix was rejected because
it matched a held-out prefix.

| Arm | Full-vocabulary top-1 / 768 | Answer choice / 768 | Both orders on changed pairs / 362 | Mean full-vocabulary NLL |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 0 | 398 | 11 | 8.1612 |
| Delta | 476 | 476 | 73 | 0.8218 |
| Memory reset | 476 | 476 | 73 | 0.6373 |
| Additive | 400 | 400 | 34 | 0.7920 |
| Decaying additive (0.5) | 538 | 538 | 107 | 0.5790 |
| Decaying additive (0.8) | 446 | 446 | 18 | 0.8049 |

At this budget the 0.5 decaying control led overall. The delta head
led the controls on seed 11 and got zero changed pairs right in both
orders on seed 29. The large NLL improvement over the frozen base
includes learning to assign probability to the two designated
answer tokens, shared by all trained arms.

This task uses a synthetic symbol vocabulary. The adapter applies its
correction at the query position. The experiment covers supervised
next-token prediction on that task, with the ordinary language
distribution and generation behavior still open.

## Consolidation scope

The source development tip is `1b35fad` and the upstream base is `4e19e15`.
A simulated Git merge found conflicts in the Poincare and Hamiltonian modules,
adapter math, export and evaluation code, public evidence docs, and generated
trainer caches. This slice transfers the standalone recurrence. Multimodal
training, vision cross-attention, the branch-specific benchmark loader, and
uncommitted legacy geometry and training changes remain in the development
checkouts. Their interfaces and evidence need review before they can be used
with the public adapter.
