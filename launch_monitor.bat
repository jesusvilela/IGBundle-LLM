@echo off
set "PYTHON_EXE=h:\LLM-MANIFOLD\igbundle-llm\unsloth_env\Scripts\python.exe"
set "SCRIPT=h:\LLM-MANIFOLD\igbundle-llm\monitor_benchmarks.py"
set "MODEL=h:\LLM-MANIFOLD\igbundle-llm\igbundle_qwen7b_Q4_K_M.gguf"

start "LLM Benchmark Monitor" "%PYTHON_EXE%" "%SCRIPT%" --model_path "%MODEL%" --benchmarks mmlu-pro aime25 gpqa arc gsm8k --limit 50
