# External benchmark harnesses

This repository does not implement or publish results for the benchmarks listed below. No score from these harnesses is claimed anywhere in this repository. The list is a starting point for future evaluations; check each project's current documentation before use.

A common approach is to serve a model behind an OpenAI-compatible endpoint (for example, llama.cpp `llama-server`) and point a third-party harness at it.

## Reasoning and general knowledge

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): MMLU-style, MMLU-Pro, and math tasks (coverage depends on configuration).
- [lighteval](https://github.com/huggingface/lighteval): MMLU-style suites, GPQA, and math task packs.
- [OpenCompass](https://github.com/open-compass/opencompass): broad benchmark collection, including GPQA and math tasks.

## Instruction following

- IFEval / IFBench.

## Agentic and long context

- SWE-bench, tau-bench, Terminal-Bench, and RULER, through their official repositories.

## ARC-AGI in this repository

`eval_arc_gguf.py` and `eval_arc.py` use the shared exact scorer in `arc_eval.py` (all test pairs, no substring matching, backend failures counted in the denominator). These benchmark entry points are not tested end-to-end with pretrained models. Separate tiny random-model checks exercise real Transformers/PEFT composition, not benchmark validity. Defaults are deliberately light (`--gpu-layers 0`, `--threads 1`):

```powershell
python eval_arc_gguf.py --model path\to\model.gguf --data-path ARC-AGI-master\data\evaluation --limit 10 --output arc_gguf_results.json
```

A valid comparison requires matched base and adapter runs under the same protocol, decoding settings, and task selection. See the [correction notice](correction-notice.md) for how historical ARC results should be read.
