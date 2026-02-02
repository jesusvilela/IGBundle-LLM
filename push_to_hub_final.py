from huggingface_hub import HfApi, upload_folder, upload_file
import os

REPO_ID = "jesusvilela/igbundle-qwen2.5-7b-riemannian"
MODEL_DIR = "output/igbundle_qwen7b_riemannian_merged"
GGUF_FILE = "igbundle_qwen7b_riemannian-unsloth.gguf"

def upload():
    api = HfApi()
    
    print(f"Creating/Verifying Repo: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, exist_ok=True)
    
    # Update README content from local root if needed
    # We assume standard unsloth save has a basic README, but we want our enhanced one.
    if os.path.exists("README.md"):
        print("Uploading Enhanced README.md...")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=REPO_ID
        )

    print(f"Uploading Merged Model Directory: {MODEL_DIR}")
    upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        ignore_patterns=["*.gguf", "checkpoint-*"] # Upload only the final merged weights
    )
    
    if os.path.exists(GGUF_FILE):
        print(f"Uploading GGUF: {GGUF_FILE}")
        api.upload_file(
            path_or_fileobj=GGUF_FILE,
            path_in_repo=GGUF_FILE,
            repo_id=REPO_ID
        )
    else:
        print(f"WARNING: GGUF file {GGUF_FILE} not found (Export might still be running). Skipping GGUF upload.")

    print("Upload Process Complete! ✅")

if __name__ == "__main__":
    upload()
