---
tags:
- geometry
- riemannian
- qwen2.5
- research
base_model: unsloth/Qwen2.5-7B-Instruct
library_name: peft
---

# IGBundle-Qwen2.5-7B-Riemannian

> **Historical research artifact.** This checkpoint has not been validated as a supported model release. The repository's current evidence does not establish improved task performance or learned hyperbolicity. Do not rely on the historical experimental claims below.

## Intended approach

IGBundle explores information-geometric operations and fiber-structured adapters as a possible inductive bias for adapting language models. The adapter design uses geometric operators and categorical components; this design intent does not demonstrate that a trained checkpoint learned a fiber bundle, semantic hierarchy, or hyperbolic representation.

## Checkpoint metadata

The original accompanying notes identify Qwen2.5-7B-Instruct as the base model and describe a LoRA plus geometric-adapter training configuration. Exact checkpoint provenance, architecture compatibility, precision, and loading procedure depend on the artifact and source snapshot. Verify those details before attempting inference.

## Evaluation status

No task-performance claim is made for this model card. The repository documents:

- A fixed-operator audit (6/6 checks) that does not verify learned geometry.
- Historical ARC-Challenge `acc_norm` of 54.86% for both an adapter checkpoint and its base model under a zero-shot run; this is not evidence of improvement.
- A September 2026 small paired token-NLL probe in which a vanilla-LoRA control outscored the geometric arm, with parameter-count and layer-coverage confounds.
- An October 2026 local subset evaluation in which the exported cp300 checkpoint did not contain the runtime geometric adapter. A reconstructed composite was not the original trained checkpoint, and its individual paired task comparisons were not statistically significant.

See the [repository evidence summary](https://github.com/jesusvilela/IGBundle-LLM/blob/main/README.md) and [October benchmark table](https://github.com/jesusvilela/IGBundle-LLM/blob/main/docs/benchmark-results/aa-index-2026-10-27.csv).

## License and use

The source repository's `LICENSE` reserves all rights. This model card does not grant permission to redistribute or use the weights; obtain authorization from the copyright holder and follow the base model's terms.
