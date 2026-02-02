
import os
from huggingface_hub import HfApi

def push_readme():
    repo_id = "jesusvilela/igbundle-qwen2.5-7b-riemannian"
    api = HfApi()
    
    print(f"Pushing fixed README.md to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj="README_model_card.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ README pushed successfully!")
    except Exception as e:
        print(f"Error pushing README: {e}")

if __name__ == "__main__":
    push_readme()
