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
the target. All three training arms start with identical weights, draw
the same minibatches and take 200 AdamW steps. Each allocates 386
parameters. The additive arm disables the erase term and its gate has
no effect on the output. The reset arm processes only the query with
zero preceding memory. Another evaluation resets memory on the trained
delta model using the same weights. Evaluation uses an independent RNG
stream from the same finite synthetic distribution.

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

| Seed | Delta | Additive | Trained with reset | Delta weights with reset |
| --- | ---: | ---: | ---: | ---: |
| 11 | 512 | 512 | 256 | 262 |
| 29 | 512 | 512 | 252 | 252 |
| 47 | 512 | 512 | 253 | 261 |

**Overwrite, correct in original order and in both orders**

| Seed | Delta / 512 | Additive / 512 | Trained with reset / 512 | Delta weights with reset / 512 | Delta on changed pairs | Additive on changed pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 512 | 385 | 258 | 246 | 254/254 | 0/254 |
| 29 | 512 | 402 | 231 | 238 | 245/245 | 0/245 |
| 47 | 512 | 404 | 257 | 254 | 231/231 | 0/231 |

The delta model learned to track the most recent write on this four-key
task. Curvature and entropy modulation were inactive in training. A
comparison with another order-sensitive mechanism, such as a decaying
additive memory, is still open. Scaling to more keys or overwrite rounds
and transfer to language modeling also remain open.

## Consolidation scope

The source development tip is `1b35fad` and the upstream base is `4e19e15`.
A simulated Git merge found conflicts in the Poincare and Hamiltonian modules,
adapter math, export and evaluation code, public evidence docs, and generated
trainer caches. This slice transfers the standalone recurrence. Multimodal
training, vision cross-attention, the branch-specific benchmark loader, and
uncommitted legacy geometry and training changes remain in the development
checkouts. Their interfaces and evidence need review before they can be used
with the public adapter.
