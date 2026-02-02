import os
import shutil
import glob

def deploy():
    TARGET_DIR = "submission_package/neurosymbolic_manifold_v3"
    # if os.path.exists(TARGET_DIR):
    #     shutil.rmtree(TARGET_DIR) # unsafe if user is in dir
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    
    print(f"Deploying to {TARGET_DIR}...")
    
    # 1. Source Code
    print("Copying Source Code...")
    shutil.copytree("src", os.path.join(TARGET_DIR, "src"), dirs_exist_ok=True)
    
    # 2. Key Scripts
    scripts = [
        "app_neural_glass.py",
        "vis_potential.py",
        "train_phase3_potential.py",
        "requirements.txt",
        "theme_neural_glass.css"
    ]
    for s in scripts:
        if os.path.exists(s):
            shutil.copy(s, TARGET_DIR)
            print(f"  Copied {s}")
        else:
            print(f"  WARNING: {s} not found!")

    # 3. Weights (Phase 3)
    weights_dir = os.path.join(TARGET_DIR, "weights")
    os.makedirs(weights_dir)
    
    weight_file = "output/phase3_adapter_potential.pt"
    if os.path.exists(weight_file):
        shutil.copy(weight_file, os.path.join(weights_dir, "adapter.pt"))
        print(f"  Copied Phase 3 Weights ({os.path.getsize(weight_file) / 1e6:.1f} MB)")
    else:
        print("  WARNING: Phase 3 Weights not found!")
        
    # 4. Create Run Script
    with open(os.path.join(TARGET_DIR, "run_demo.bat"), "w") as f:
        f.write("@echo off\n")
        f.write("echo Starting Neural Glass...\n")
        f.write("python app_neural_glass.py\n")
        f.write("pause\n")
        
    print("Deployment Complete.")

if __name__ == "__main__":
    deploy()
