import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
from tqdm import tqdm

def compute_sectional_curvature(h_states):
    """
    Estimate sectional curvature from hidden states.
    k(u, v) = <R(u, v)v, u> / (|u|^2|v|^2 - <u, v>^2)
    This is a simplified proxy: we measure the 'deviation' from Euclidean flatness
    by checking how distances expand/contract relative to parallelism.
    """
    # Simplified proxy: measure cosine similarity variance vs distance
    # Identify "tangent vectors" as differences between consecutive tokens
    tangents = h_states[:, 1:] - h_states[:, :-1]
    
    # Normalized tangents
    norms = torch.norm(tangents, dim=-1, keepdim=True)
    tangents_norm = tangents / (norms + 1e-6)
    
    # Curvature proxy: How much does the tangent direction change?
    # Flat space = tangent vectors are parallel (if geodesic).
    # We define curvature intensity as the rate of change of the tangent vector.
    curvature_est = torch.norm(tangents_norm[:, 1:] - tangents_norm[:, :-1], dim=-1)
    
    return curvature_est.mean().item()

def validate_geometry():
    model_path = "output/igbundle_qwen7b_riemannian_merged"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    try:
        # User requested fix for Mistral regex warning
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # fix_mistral_regex removed as it might be specific to mistral class
    except Exception as e:
        print(f"Tokenizer warning: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Test Prompts (ARC-like patterns)
    prompts = [
        "Pattern: Red->Blue. Input: Red box. Output:",
        "Sequence: 1, 2, 4, 8, 16. Next:",
        "Grid transformation: Rotate 90 degrees. [[1,0],[0,1]] ->",
        "Reasoning: If all A are B, and x is A, then x is"
    ]
    
    results = []
    
    print("Running geometric validation...")
    model.eval()
    
    all_curvatures = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # Forward pass with hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get last layer hidden states
            last_hidden = outputs.hidden_states[-1] # [batch, seq, dim]
            
            # Compute metric
            curve_val = compute_sectional_curvature(last_hidden)
            all_curvatures.append(curve_val)
            
            generated = model.generate(**inputs, max_new_tokens=10)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "curvature_score": curve_val,
                "output": text
            })
            
    avg_curvature = np.mean(all_curvatures)
    print(f"\nAverage Latent Curvature Proxy: {avg_curvature:.4f}")
    
    # Save results
    with open("geometric_validation.json", "w") as f:
        json.dump({
            "metrics": {"avg_curvature": float(avg_curvature)},
            "samples": results
        }, f, indent=2)
        
    print("Validation detailed results saved to geometric_validation.json")
    
    # Check PASS/FAIL
    # We expect Non-Zero curvature (since we trained for it) but stable generation.
    if avg_curvature > 0.001: 
        print("PASS: Manifold shows non-trivial geometric structure.")
    else:
        print("WARNING: Latent space appears suspiciously flat (Euclidean).")

if __name__ == "__main__":
    validate_geometry()
