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

> **Historical research artifact.** This checkpoint has not been validated as a supported model release. The repository's current evidence does not establish improved task performance or learned hyperbolicity. Do not rely on historical experimental claims.

## Intended approach

IGBundle explores information-geometric operations and fiber-structured adapters as a possible inductive bias for adapting language models. The adapter design uses geometric operators and categorical components; this design intent does not demonstrate that a trained checkpoint learned a fiber bundle, semantic hierarchy, or hyperbolic representation.

## Checkpoint metadata

The original accompanying notes identify Qwen2.5-7B-Instruct as the base model and describe a LoRA plus geometric-adapter training configuration. Exact checkpoint provenance, architecture compatibility, precision, and loading procedure depend on the artifact and source snapshot. Verify those details before attempting inference.

## Evaluation status

The [expanded export/image audit](https://github.com/jesusvilela/IGBundle-LLM/blob/main/docs/export-image-audit.md) inspected two distinct local GGUFs, not necessarily this card's checkpoint. Both expose standard Qwen2 tensor names; their inspected fast runner does not restore the separate geometric adapter. Neither the file names nor the archived visualizations establish a hyperbolic or hypercomplex inference architecture.

No task-performance claim is made for this model card:

- A historical audit reports 6/6 checks, but lacks immutable source/weight binding and uses heuristic proxy criteria. It is not an operator certification or evidence about this checkpoint.
- Historical ARC-Challenge zero-shot `acc_norm` was 54.86% for both an adapter checkpoint and its base model.
- A September 2026 small paired token-NLL probe favored its vanilla-LoRA control, with parameter-count and layer-coverage confounds.
- An October 2026 local subset evaluation found no significant individual paired task differences. The exported cp300 checkpoint did not contain the runtime geometric adapter; its reconstructed composite was not the original trained checkpoint.

See the [repository evidence summary](https://github.com/jesusvilela/IGBundle-LLM/blob/main/README.md) and [correction notice](https://github.com/jesusvilela/IGBundle-LLM/blob/main/docs/correction-notice.md).

## License and use

The source repository's `LICENSE` reserves all rights. This card does not grant permission to redistribute or use the weights; obtain authorization from the copyright holder and follow the base model's terms.
