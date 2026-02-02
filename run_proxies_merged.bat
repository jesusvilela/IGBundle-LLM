@echo off
set HF_HOME=H:\hf_cache
set TMP=H:\tmp
set TEMP=H:\tmp
echo Starting Proxy Benchmarks (HellaSwag, TruthfulQA) on Merged Model...
python -m lm_eval --model hf --model_args pretrained=output/igbundle_qwen7b_riemannian_merged,trust_remote_code=True,dtype=bfloat16 --tasks hellaswag,truthfulqa_mc2 --batch_size 1 --device cuda:0 --output_path eval_results_proxies_merged --log_samples
echo Done.
