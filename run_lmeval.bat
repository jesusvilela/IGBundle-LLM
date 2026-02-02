@echo off
set HF_HOME=H:\hf_cache
set TMP=H:\tmp
set TEMP=H:\tmp
echo Starting lm-eval on arc_challenge...
python -m lm_eval --model hf --model_args pretrained=output/igbundle_qwen7b_riemannian_merged,trust_remote_code=True,dtype=bfloat16 --tasks arc_challenge --batch_size 1 --device cuda:0 --output_path eval_results_lmeval --log_samples
echo Done.
