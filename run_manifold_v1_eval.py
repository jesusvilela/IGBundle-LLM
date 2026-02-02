
import os
import sys
import torch
import json
import logging
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from collections import defaultdict

sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter

# Setup Output
OUTPUT_FILE = "manifold_v1_report.md"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManifoldEval")

# Config
MODEL_ID = "../igbundle_qwen7b_cp600"
# Try to find latest checkpoint from diverse training, or fallback to symmetry/unified
def find_latest_checkpoint():
    # Check diverse training first
    diverse_dir = "igbundle_diverse_training"
    if os.path.exists(diverse_dir):
        checkpoints = sorted([d for d in os.listdir(diverse_dir) if d.startswith("checkpoint-")], 
                             key=lambda x: int(x.split("-")[1]), reverse=True)
        if checkpoints:
            return os.path.join(diverse_dir, checkpoints[0], "adapter_weights.pt")
    
    # Fallback
    sym_dir = "igbundle_symmetry_training"
    if os.path.exists(sym_dir):
        checkpoints = sorted([d for d in os.listdir(sym_dir) if d.startswith("checkpoint-")], 
                             key=lambda x: int(x.split("-")[1]), reverse=True)
        if checkpoints:
            return os.path.join(sym_dir, checkpoints[0], "adapter_weights.pt")
            
    return None

ADAPTER_PATH = find_latest_checkpoint()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAINS = {
    "Logic": [
        "Solve for x: 3x + 5 = 20",
        "If all men are mortal and Socrates is a man, what follows?",
        "Write a proof that the sum of two odd numbers is even."
    ],
    "Creative": [
        "Write a poem about a lost spaceship.",
        "Describe a color that doesn't exist.",
        "Invent a creation myth for a digital universe."
    ]
}

class ManifoldEvaluator:
    def __init__(self):
        self.tokenizer = None
        self.llm = None
        self.adapter = None
        self.geo_state_capture = None
    
    def load(self):
        logger.info(f"Loading Base Model: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        
        # Load Adapter
        logger.info(f"Loading Adapter from {ADAPTER_PATH}")
        config = IGBundleConfig(
            hidden_size=3584,
            num_components=8,
            latent_dim=64,
            num_categories=16,
            use_dynamics=True,
            use_geodesic_attn=True,
            supported_modalities=["vision", "text"]
        )
        self.adapter = create_geometric_adapter(config).to(DEVICE)
        
        if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
            self.adapter.load_state_dict(torch.load(ADAPTER_PATH))
            logger.info("Adapter weights loaded.")
        else:
            logger.warning("No adapter weights found! Running initialized weights.")
            
        self._inject_hook()
        
    def _inject_hook(self):
        target = self.llm.model.layers[12]
        orig_fwd = target.forward
        
        def hook(hidden_states, *args, **kwargs):
            out = orig_fwd(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            orig_dtype = h.dtype
            h_in = h.to(DEVICE).float()
            
            # Forward Adapter
            adapted_h, geo_state = self.adapter(h_in)
            self.geo_state_capture = geo_state # Capture last state
            
            adapted = adapted_h.to(orig_dtype)
            return (adapted,) + out[1:] if isinstance(out, tuple) else adapted
            
        target.forward = hook
        logger.info("Hook injected.")
        
    def run_tests(self):
        results = []
        
        logger.info("Starting Evaluation Loop...")
        
        with open(OUTPUT_FILE, "w") as f:
            f.write("# ManifoldGL V1 Evaluation Report\n\n")
            f.write(f"**Adapter**: `{ADAPTER_PATH}`\n")
            f.write(f"**Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
        # 1. Stability Test
        # Run same prompt 3 times
        seed_prompt = DOMAINS["Logic"][0]
        indices_history = []
        
        for i in range(3):
            logger.info(f"Stability Run {i+1}...")
            inputs = self.tokenizer(seed_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.llm.generate(**inputs, max_new_tokens=20)
                
            if self.geo_state_capture.active_indices is not None:
                idx = self.geo_state_capture.active_indices[0].tolist() # (k,)
                indices_history.append(set(idx))
            else:
                indices_history.append(set())
                
        # Calculate Jaccard Similarity between runs
        stability_score = 0.0
        if len(indices_history) > 1:
            u = indices_history[0].union(indices_history[1])
            i = indices_history[0].intersection(indices_history[1])
            if len(u) > 0:
                stability_score = len(i) / len(u)
        
        with open(OUTPUT_FILE, "a") as f:
            f.write("## 1. Stability Analysis\n")
            f.write(f"- **Prompt**: '{seed_prompt}'\n")
            f.write(f"- **Active Indices**: {indices_history}\n")
            f.write(f"- **Stability Score (Jaccard)**: {stability_score:.2f} (1.0 = Perfect Invariance)\n\n")

        # 2. Differentiation Test (Logic vs Creative)
        logic_indices = set()
        creative_indices = set()
        
        # Logic
        logger.info("Testing Logic Domain...")
        inputs = self.tokenizer(DOMAINS["Logic"][1], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            self.llm.generate(**inputs, max_new_tokens=20)
        if self.geo_state_capture.active_indices is not None:
             logic_indices = set(self.geo_state_capture.active_indices[0].tolist())

        # Creative
        logger.info("Testing Creative Domain...")
        inputs = self.tokenizer(DOMAINS["Creative"][0], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            self.llm.generate(**inputs, max_new_tokens=20)
        if self.geo_state_capture.active_indices is not None:
             creative_indices = set(self.geo_state_capture.active_indices[0].tolist())
             
        # Overlap
        overlap = logic_indices.intersection(creative_indices)
        
        with open(OUTPUT_FILE, "a") as f:
             f.write("## 2. Domain Differentiation\n")
             f.write(f"- **Logic Fiber Locus**: {logic_indices}\n")
             f.write(f"- **Creative Fiber Locus**: {creative_indices}\n")
             f.write(f"- **Overlap**: {overlap}\n")
             if logic_indices != creative_indices:
                 f.write("- **Result**: PASS (Distinct Activations Detected)\n\n")
             else:
                 f.write("- **Result**: FAIL (Mode Collapse / No Differentiation)\n\n")

        logger.info("Evaluation Complete. Report written.")

if __name__ == "__main__":
    evaluator = ManifoldEvaluator()
    evaluator.load()
    evaluator.run_tests()
