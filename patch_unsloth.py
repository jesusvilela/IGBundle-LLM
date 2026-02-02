import os
import site
import glob

def patch_unsloth():
    # Find unsloth_zoo location
    paths = site.getsitepackages()
    target_file = None
    
    for p in paths:
        search_path = os.path.join(p, "unsloth_zoo", "fused_losses", "cross_entropy_loss.py")
        if os.path.exists(search_path):
            target_file = search_path
            break
            
    if not target_file:
        # Try local venv path assumption
        base = os.path.dirname(__file__)
        # Assuming we run from h:\LLM-MANIFOLD\igbundle-llm
        search_path = os.path.join(base, "unsloth_env", "Lib", "site-packages", "unsloth_zoo", "fused_losses", "cross_entropy_loss.py")
        if os.path.exists(search_path):
             target_file = search_path
             
    if not target_file:
        print("Could not find unsloth_zoo/fused_losses/cross_entropy_loss.py")
        return

    print(f"Patching {target_file}...")
    
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Patch logic
    # Look for "target_gb ="
    # Depending on version, it might be inside _get_chunk_multiplier
    
    if "target_gb = max(target_gb, 0.1)" in content:
        print("Already patched.")
        return

    # Naive replace: we want to prevent division by zero.
    # The traceback showed: "multiplier = ... / (target_gb)"
    # We will search for where target_gb is assigned or used.
    
    # Check if we can just wrap the division?
    # Better: inject a safety max() after calculation.
    
    # Standard Unsloth code often has: 
    # target_gb = free_gpu_memory ...
    
    # Let's verify content first? No, blind patch is risky.
    # But I can read it via python print!
    
    # PRINT CONTENT FIRST
    start = content.find("_get_chunk_multiplier")
    if start != -1:
         sub = content[start:start+500]
         # print("Context:", sub)
         
         # The crash line is likely: multiplier = (vocab_size * 4 / ...) / (target_gb)
         # We want to change it to / (target_gb + 1e-6) or fix target_gb.
         
         if "/ (target_gb)" in content:
             new_content = content.replace("/ (target_gb)", "/ (max(target_gb, 0.1))")
             
             with open(target_file, 'w', encoding='utf-8') as f:
                 f.write(new_content)
             print("Patched successfully.")
         else:
             print("Could not find exact division pattern '/ (target_gb)'. Manual check needed.")
             print(sub) # Print for debug
    else:
         print("Function _get_chunk_multiplier not found.")

if __name__ == "__main__":
    patch_unsloth()
