@echo off
set HF_HOME=H:\hf_cache
set TMP=H:\tmp
set TEMP=H:\tmp
echo Starting TruthfulQA on BASELINE Model...
python -m lm_eval --model hf --model_args pretrained=unsloth/Qwen2.5-7B-Instruct,trust_remote_code=True,dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --device cuda:0 --output_path eval_results_tqa_baseline --log_samples
echo Done.
