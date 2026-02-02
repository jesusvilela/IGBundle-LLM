@echo off
REM Activate Virtual Environment
call "h:\LLM-MANIFOLD\igbundle-llm\unsloth_env\Scripts\activate.bat"

set HF_HOME=H:\hf_cache
set TMP=H:\tmp
set TEMP=H:\tmp

echo ===================================================
echo Starting OPTIMIZED Benchmark Suite on Merged Model
echo Tasks: arc_challenge, truthfulqa_mc2, winogrande
echo Mode: Loglikelihood (Fast)
echo ===================================================

python -m lm_eval --model hf --model_args pretrained=output/igbundle_qwen7b_riemannian_merged,trust_remote_code=True,dtype=bfloat16 --tasks arc_challenge,truthfulqa_mc2,winogrande --batch_size 1 --device cuda:0 --output_path eval_results_opt_merged --log_samples

echo ===================================================
echo Benchmark Complete
echo ===================================================
