@echo off
REM Activate Virtual Environment
call "h:\LLM-MANIFOLD\igbundle-llm\unsloth_env\Scripts\activate.bat"

set HF_HOME=H:\hf_cache
set TMP=H:\tmp
set TEMP=H:\tmp

echo ===================================================
echo Starting GSM8K Benchmark (4-bit Quantization)
echo Strategy: Quantize to fit 8GB VRAM and avoid swapping
echo ===================================================

python -m lm_eval --model hf --model_args pretrained=output/igbundle_qwen7b_riemannian_merged,trust_remote_code=True,load_in_4bit=True --tasks gsm8k --batch_size 1 --device cuda:0 --output_path eval_results_gsm8k --log_samples

echo ===================================================
echo Benchmark Complete
echo ===================================================
