@echo off
set HF_HOME=H:\hf_cache
set TMP=H:\tmp
set TEMP=H:\tmp
echo Starting TruthfulQA on Merged Model (Fixed Config)...
python -m lm_eval --model hf --model_args pretrained=output/igbundle_qwen7b_riemannian_merged,trust_remote_code=True,dtype=bfloat16,fix_mistral_regex=True --tasks truthfulqa_mc2 --batch_size 1 --device cuda:0 --output_path eval_results_tqa_merged --log_samples
echo Done.
