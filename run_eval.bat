@echo off
set HF_HOME=H:\hf_cache
python -m lm_eval --model hf --model_args pretrained=output/igbundle_qwen7b_riemannian_merged,trust_remote_code=True --tasks arc_challenge --device cuda:0 --batch_size 1 > eval_results.txt
type eval_results.txt
