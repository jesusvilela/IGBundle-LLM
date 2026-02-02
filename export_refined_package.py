import os
import shutil
import subprocess
import sys
import torch

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Assumes we are in h:\LLM-MANIFOLD\igbundle-llm
DIST_DIR = os.path.join(BASE_DIR, "dist")
SRC_DIR = os.path.join(BASE_DIR, "src", "igbundle")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "igbundle_refined_training", "checkpoint-100")
ADAPTER_WEIGHTS_SOURCE = os.path.join(CHECKPOINT_DIR, "adapter_weights.pt")
LLAMA_CPP_DIR = os.path.join(os.path.dirname(BASE_DIR), "llama.cpp") # h:\LLM-MANIFOLD\llama.cpp
MODEL_SOURCE_HF = os.path.abspath(os.path.join(BASE_DIR, "..", "igbundle_qwen7b_cp600")) # Adjust if needed

def main():
    print("Starting Export Process...")
    
    # 1. Create Dist Dir
    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    os.makedirs(DIST_DIR)
    print(f"Created dist directory at {DIST_DIR}")
    
    # 2. Copy Source Code (for Refined Mode)
    # We copy 'src/igbundle' to 'dist/igbundle_lib/igbundle'
    # Wait, the inference script expects 'igbundle' to be importable from 'igbundle_lib'.
    # So we copy the contents of 'src' into 'dist/igbundle_lib'?
    # No, 'src/igbundle' -> 'dist/igbundle_lib/igbundle'
    target_lib = os.path.join(DIST_DIR, "igbundle_lib", "igbundle")
    shutil.copytree(SRC_DIR, target_lib)
    print("Copied source code.")
    
    # 3. Copy Adapter Weights
    if os.path.exists(ADAPTER_WEIGHTS_SOURCE):
        shutil.copy(ADAPTER_WEIGHTS_SOURCE, os.path.join(DIST_DIR, "adapter_refined.pt"))
        print(f"Copied adapter weights from {ADAPTER_WEIGHTS_SOURCE}")
    else:
        print(f"WARNING: Adapter weights not found at {ADAPTER_WEIGHTS_SOURCE}. Please train first.")
        
    # 4. Copy Inference Script
    # We read the script we just created in src/igbundle/scripts and write it to dist
    script_source = os.path.join(BASE_DIR, "src", "igbundle", "scripts", "run_inference_integrated.py")
    shutil.copy(script_source, os.path.join(DIST_DIR, "run_inference_integrated.py"))
    print("Copied inference script.")
    
    # 5. GGUF Conversion
    # We need to run llama.cpp/convert_hf_to_gguf.py
    # Command: python convert_hf_to_gguf.py <model_dir> --outfile <dist>/base_model.gguf --outtype q4_0
    
    gguf_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if not os.path.exists(gguf_script):
        print(f"WARNING: GGUF conversion script not found at {gguf_script}.")
        print("Skipping GGUF generation. You must generate 'base_model.gguf' manually.")
    else:
        print("Starting GGUF Conversion (this may take a while)...")
        output_gguf = os.path.join(DIST_DIR, "base_model.gguf")
        
        # Check input model
        if not os.path.exists(MODEL_SOURCE_HF):
             print(f"Model source {MODEL_SOURCE_HF} not found. Cannot convert.")
        else:
            try:
                cmd = [
                    sys.executable,
                    gguf_script,
                    MODEL_SOURCE_HF,
                    "--outfile", output_gguf,
                    "--outtype", "q8_0"
                ]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                print("GGUF Conversion Complete.")
            except subprocess.CalledProcessError as e:
                print(f"GGUF Conversion Failed: {e}")
                
    print("\n-------------------------------------------")
    print(f"Export Complete! Package available at: {DIST_DIR}")
    print("To run:")
    print(f"  cd {DIST_DIR}")
    print("  python run_inference_integrated.py --mode fast --prompt 'Hello'")
    print("-------------------------------------------")

if __name__ == "__main__":
    main()
