import os
import shutil
import time

SOURCE_DIR = "igbundle_unified_training/final"
DEPLOY_DIR = "igbundle_unified_deployment"
SRC_CODE_DIR = "src"

def export():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist. Run training first.")
        return

    print(f"Exporting Unified Model to {DEPLOY_DIR}...")
    
    # 1. Create Directory
    if os.path.exists(DEPLOY_DIR):
        shutil.rmtree(DEPLOY_DIR)
    os.makedirs(DEPLOY_DIR)
    
    # 2. Copy Weights
    src_weights = os.path.join(SOURCE_DIR, "adapter_weights.pt")
    dst_weights = os.path.join(DEPLOY_DIR, "adapter_weights.pt")
    if os.path.exists(src_weights):
        shutil.copy2(src_weights, dst_weights)
        print(f"Copied weights: {dst_weights}")
    else:
        print(f"Error: Weights not found at {src_weights}")
        
    # 3. Copy Source Code
    dst_src = os.path.join(DEPLOY_DIR, "src")
    shutil.copytree(SRC_CODE_DIR, dst_src)
    print(f"Copied source code to {dst_src}")
    
    # 4. Copy Inference Script
    shutil.copy2("run_unified_inference.py", os.path.join(DEPLOY_DIR, "run_inference.py"))
    print("Copied inference script.")
    
    # 5. Create Readme
    with open(os.path.join(DEPLOY_DIR, "README.md"), "w") as f:
        f.write("# Unified Manifold Model Deployment\n\n")
        f.write("## Usage\n")
        f.write("```bash\n")
        f.write("python run_inference.py --text 'Explain quantum mechanics.'\n")
        f.write("python run_inference.py --image 'path/to/image.jpg' --text 'Describe this.'\n")
        f.write("```\n\n")
        f.write("## Requirements\n")
        f.write("- transformers\n- torch\n- bitsandbytes\n- accelerate\n- pillow\n")
        
    print("Export Complete.")

if __name__ == "__main__":
    export()
