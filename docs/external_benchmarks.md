# External benchmark harnesses (common)

This repo does not implement the full benchmark set you listed. The usual approach is to run the model behind an OpenAI-compatible API (llama.cpp `llama-server`) and point third-party harnesses at it.

I cannot browse the web from this environment, but the packages below are commonly used in public LLM evals and accept OpenAI-compatible endpoints.

## Reasoning / General knowledge
- lighteval: MMLU-style suites, GPQA, AIME-style math, MiniF2F (depending on task packs)
- lm-evaluation-harness: broad coverage of MMLU/MMLU-Pro-like subsets and math tasks (varies by config)
- OpenCompass: large benchmark zoo, including GPQA and math tasks
- bigcode-evaluation-harness: LiveCodeBench-style code evals

## Agentic
- SWE-bench (OpenHands) for SWE-Bench
- TauBench v2 (official repo)
- Terminal Bench (official repo)

## Chat / Instruction
- IFEval / IFBench (prompt-following)
- LMSYS Arena-Hard v2 (official scripts)
- Scale AI Multi Challenge (requires access)

## Long context
- RULER (official repo)
- AA-LCR (if you have the dataset release)

## Multilingual
- MMLU-ProX (official repo)
- WMT24++ (official WMT release)

## Minimal llama.cpp server setup
1) Start the server:
   - scripts/run_llama_server.ps1 -ModelPath igbundle_qwen7b.gguf
2) Export OpenAI-compatible env vars:
   - scripts/set_eval_env.ps1
3) Configure your chosen harness to use `OPENAI_API_BASE` and `OPENAI_API_KEY`.

If you want, tell me which specific harness you plan to run first and I will add a launcher for it.
