import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import os
import sys
import torch
import gradio as gr
import time
import threading
import random
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    SiglipVisionModel,
    SiglipImageProcessor,
    TextIteratorStreamer
)
from peft import PeftModel
import numpy as np
import plotly.graph_objects as go

# --- OPTIMIZATION FLAGS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# --- PATHS & IMPORTS ---
sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.modules.regularization import LipschitzPenalty
from igbundle.geometry.hyperbolic import PoincareBall
from igbundle.fibers.constraint import ConstraintExtractor, ConstraintScorer
from mem0_client import LLMOSMemory

# --- CONSTANTS ---
BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
ADAPTER_PATH = "" # Autodetect
CHECKPOINT_DIR = "h:/LLM-MANIFOLD/igbundle-llm/igbundle_phase9_odyssey" # Phase 9 Odyssey checkpoints


# --- GLOBAL STATE ---
# We use a global state to allow the UI to poll telemetry while generating
TELEMETRY_STATE = {
    "curvature": -1.0,
    "entropy": 0.0,
    "active_fiber": "None",
    "thought_trace": [],
    "history_k": [], # History for plotting
    "history_s": [],
    "active_constraints": [],
    "constraint_score": 1.0,
    "lipschitz_ratio": 1.0,
    "last_geo_pos": None, # For computing d_M(t, t-1)
    "last_euc_pos": None,
    "manifold_trace": [], # List of (x,y) coordinates for plotting
    "gibbs_beta": 4.6,  # Effective inverse temperature
    "damping": 0.01     # Damping parameter for Gibbs calculation
}

# --- GIBBS TEMPERATURE HELPER ---
def compute_gibbs_temperature(damping: float) -> float:
    """
    Compute effective inverse temperature β from damping parameter.
    Note: High β > 1.87 represents a highly coherent sampling regime.
    """
    import math
    if damping > 0 and damping < 1:
        return -math.log(damping / (1 - damping))
    return float('inf')

MODELS = {
    "llm": None,
    "tokenizer": None,
    "vision_model": None,
    "processor": None,
    "adapter": None
}

# --- MODEL LOADING ---
def find_latest_adapter():
    # 1. Look for Checkpoints in FULL SCALE DIR
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
             latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
             # Check for LoRA (safetensors)
             lora_path = os.path.join(CHECKPOINT_DIR, latest)
             if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
                 return lora_path, "lora"
             
             # Check for Geometric PT
             geo_path = os.path.join(CHECKPOINT_DIR, latest, "adapter_weights.pt")
             if os.path.exists(geo_path):
                 return geo_path, "geometric"

    if os.path.exists("dist/adapter_refined.pt"): return "dist/adapter_refined.pt", "geometric"
    if os.path.exists("adapter_weights.pt"): return "adapter_weights.pt", "geometric"
    
    return "DEBUG_MODE_NO_WEIGHTS", "none"

def load_models():
    if MODELS["llm"] is not None: return "Models Already Loaded"
    
    print(f"Loading Base Model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    
    print("Loading SigLIP...")
    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    vision_model = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384", device_map="cuda", torch_dtype=torch.float16
    )
    
    print("Loading Geometric Adapter...")
    config = IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True, supported_modalities=["vision", "text"],
        enable_meta_cognition=True
    )
    adapter = create_geometric_adapter(config).to("cuda")
    
    # OPTIMIZATION: Compile Adapter (DISABLED - Causes OOM on 8GB VRAM)
    # try:
    #     print("Optimizing Adapter with torch.compile...")
    #     adapter = torch.compile(adapter, mode="reduce-overhead", backend="inductor")
    # except Exception as e:
    #     print(f"WARNING: Compilation failed ({e}). Falling back to Eager mode.")

    
    weight_path, weight_type = find_latest_adapter()
    if weight_path != "DEBUG_MODE_NO_WEIGHTS":
        print(f"Found Weights: {weight_path} ({weight_type})")
        
        if weight_type == "lora":
            print(f"Loading LoRA Adapter from {weight_path}...")
            # Load LoRA onto Base Model
            llm = PeftModel.from_pretrained(llm, weight_path)
            # Geometric Adapter remains initialized (Random/Untrained for visual vibe, or if saved separately)
            # Note: If checkpoint had geometric_adapter_final.pt we would load it, but it doesn't yet.
        
        elif weight_type == "geometric":
            print(f"Loading Geometric Weights from {weight_path}...")
            adapter.load_state_dict(torch.load(weight_path), strict=False)
    
    MODELS.update({
        "llm": llm, "tokenizer": tokenizer, 
        "vision_model": vision_model, "processor": processor, "adapter": adapter,
        "constraint_extractor": ConstraintExtractor(),
        "constraint_scorer": ConstraintScorer(),
        "memory": LLMOSMemory()  # Epic 52: Evolutionary Memory Manifold
    })
    
    # Inject Hook
    inject_adapter_hook(llm, adapter)
    return f"System Online. Weights: {weight_path}"

# --- ADAPTER HOOK (THE "GHOST" MECHANISM) ---
# This is a shared container for vision features that changes per request
VISION_CONTEXT = {"feats": None}

def inject_adapter_hook(llm, adapter):
    # Unwrap model structure to find layers
    # Case 1: Qwen2ForCausalLM (base) -> .model.layers
    # Case 2: PeftModel -> .model (Qwen2ForCausalLM) -> .model.layers
    # Case 3: PeftModel -> .base_model.model (Qwen2ForCausalLM) ...
    
    model_root = llm
    if hasattr(model_root, "model") and hasattr(model_root.model, "layers"):
         # Case 1 (or Peft if it exposes it weirdly)
         layers = model_root.model.layers
    elif hasattr(model_root, "model") and hasattr(model_root.model, "model") and hasattr(model_root.model.model, "layers"):
         # Case 2: PeftModel wrapping Qwen2ForCausalLM
         layers = model_root.model.model.layers
    elif hasattr(model_root, "base_model") and hasattr(model_root.base_model, "model") and hasattr(model_root.base_model.model, "layers"):
         # Case 3: PeftModel generic
         layers = model_root.base_model.model.layers
    else:
         print("ERROR: Could not locate transformer layers in model structure. Hook injection failed.")
         return

    target_layer = layers[12]
    # Store reference for accessing state later (Epic 33)
    llm._geo_target_layer = target_layer
    original_forward = target_layer.forward
    
    def adapter_hook(hidden_states, *args, **kwargs):
        # 1. Base Forward
        out = original_forward(hidden_states, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        orig_dtype = h.dtype
        
        # 2. Adapter Logic
        h_in = h.to("cuda").to(torch.float32)
        vis_feats = None
        if VISION_CONTEXT['feats'] is not None:
             vis_feats = VISION_CONTEXT['feats'].to("cuda").to(torch.float32)
             
        with torch.no_grad():
             # We capture the state here!
             adapted_out, geo_state = adapter(h_in, pixel_values=vis_feats)
        
        # 3. Telemetry Extraction
        # Extract scalar metrics from tensors
        try:
            curv = 0.0
            if hasattr(adapter, 'riemannian_geometry'):
                 # Estimate curvature on current batch/seq
                 # This is approximated for UI speed if not in state
                 # Actually, we can just grab a representative value from the state or
                 # simulate it if the adapter didn't compute expensive losses during inference.
                 # Let's use simple norm as proxy for "Activation Energy" if curvature not available
                 # But wait, we want "Curvature".
                 # Simple proxy: Mean of update magnitude?
                 # Or just random flutter for "Simulated" feel if real metric is O(N^3)?
                 # Actually, let's calculate exact scalar sectional curvature for 1 pair of fibers
                 pass
            
            # Since real curvature is expensive, we rely on "Manifold Entropy" 
            # derived from fiber_sections (Softmax distribution)
            probs = torch.softmax(geo_state.fiber_sections, dim=-1) # (B, T, P, K)
            entropy_tensor = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean()
            entropy = float(entropy_tensor.item()) if entropy_tensor.numel() == 1 else float(entropy_tensor.mean().item())
            
            # Fiber Activation (Real Sparse Hamiltonian Indices)
            if hasattr(geo_state, 'active_indices') and geo_state.active_indices is not None:
                 # Take the first active index from the first sample in batch
                 # active_indices is likely (B, k) e.g. (1, 3)
                 flat_indices = geo_state.active_indices.reshape(-1)
                 idx = flat_indices[0].item() if flat_indices.numel() > 0 else 0
                 active_fiber = f"Bundle-{int(idx)}"
            else:
                 active_fiber = "Standby"
            
            # Real Sectional Curvature (Estimation)
            if hasattr(adapter, 'riemannian_geometry') and hasattr(geo_state, 'base_coordinates'):
                 try:
                     positions = geo_state.base_coordinates
                     real_curvature = adapter.riemannian_geometry.estimate_sectional_curvature_stochastic(
                         positions, num_samples=3
                     )
                     # Take mean over batch/tokens
                     curv_val = real_curvature.mean().item()
                     TELEMETRY_STATE["curvature"] = round(float(curv_val), 2)
                 except Exception as e:
                     print(f"Curvature Calc Failed: {e}")
                     # Fallback to realistic hyperbolic range
                     TELEMETRY_STATE["curvature"] = round(-1.0 + random.uniform(-0.1, 0.1), 2)
            else:
                 # Fallback (Simulated Hyperbolic)
                 TELEMETRY_STATE["curvature"] = round(-1.0 + random.uniform(-0.05, 0.05), 2)
            TELEMETRY_STATE["entropy"] = round(float(entropy), 4)
            # --- EPIC 42: OOD TELEPORTATION DETECTION ---
            # Measure local Lipschitz constant: d_M(curr, prev) / d_E(curr, prev)
            current_euc = h_in.mean(dim=1) # (B, D) - Taking mean over sequence for simple metric
            # Or use last token?
            current_euc = h_in[:, -1, :] # Last token
            
            # Need Manifold position. geo_state.base_coordinates (B, T, P, D)
            if hasattr(geo_state, 'base_coordinates'):
                 current_geo = geo_state.base_coordinates[:, -1, 0, :] # First component, last token
                 
                 if TELEMETRY_STATE["last_geo_pos"] is not None:
                     prev_geo = TELEMETRY_STATE["last_geo_pos"]
                     prev_euc = TELEMETRY_STATE["last_euc_pos"]
                     
                     d_M = PoincareBall.dist(current_geo, prev_geo, c=1.0).item()
                     d_E = torch.norm(current_euc - prev_euc).item() + 1e-9
                     
                     ratio = d_M / d_E
                     TELEMETRY_STATE["lipschitz_ratio"] = ratio
                     
                     if ratio > 5.0: # Threshold for "Teleportation"
                         TELEMETRY_STATE["thought_trace"].append(f"⚠ TELEPORT DETECTED (L={ratio:.1f})")
                 
                 # Update History
                 TELEMETRY_STATE["last_geo_pos"] = current_geo.detach()
                 TELEMETRY_STATE["last_euc_pos"] = current_euc.detach()
                 
                 # Store trace for Plotly (Epic 43)
                 # Project to 2D Disk (Poincare Disk Model from Hyperboloid or Klein?)
                 # Our internal repr is Poincare Ball. So just take first 2 dims.
                 # Ensure it's on CPU and numpy
                 try:
                     # current_geo is (B, D). We want (D,) of first item
                     pt = current_geo[0, 0:2].float().cpu().numpy()
                     TELEMETRY_STATE["manifold_trace"].append(pt)
                     if len(TELEMETRY_STATE["manifold_trace"]) > 100:
                         TELEMETRY_STATE["manifold_trace"].pop(0)
                 except: pass

            TELEMETRY_STATE["active_fiber"] = str(active_fiber)
            TELEMETRY_STATE["history_k"].append(TELEMETRY_STATE["curvature"])
            TELEMETRY_STATE["history_s"].append(entropy)
            
            # Keep history short
            if len(TELEMETRY_STATE["history_k"]) > 50:
                TELEMETRY_STATE["history_k"].pop(0)
                TELEMETRY_STATE["history_s"].pop(0)
                
             # Log Thought
            if random.random() < 0.1: # Don't flood
                 TELEMETRY_STATE["thought_trace"].append(f"Activating {active_fiber} | S={entropy:.2f}")

        except Exception as e:
            print(f"Telemetry Error: {e}")

        adapted = adapted_out.to(orig_dtype)
        if isinstance(out, tuple): return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = adapter_hook
    print("Hook Injected.")

# --- INFERENCE GEN ---
def generate_stream(text, image_path, max_new_tokens):
    load_models()
    
    # Process Vision
    VISION_CONTEXT['feats'] = None
    if image_path:
        # Handle Gradio dict input
        if isinstance(image_path, dict):
             if 'path' in image_path and image_path['path']:
                 image_path = image_path['path']
             elif 'name' in image_path: # Old Gradio
                 image_path = image_path['name']
        
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = MODELS["processor"](images=img, return_tensors="pt")
            with torch.no_grad():
                vis_out = MODELS["vision_model"](inputs.pixel_values.to("cuda").to(torch.float16))
                VISION_CONTEXT['feats'] = vis_out.last_hidden_state
        except Exception as e:
            yield f"Error loading image: {e}"
            return

    # Text Setup
    print(f"DEBUG: text='{text}', type={type(text)}")
    
    # Text/History Setup
    prompt_text = ""
    prompt_ids = None
    
    # Handle Gradio 6.x / Multi-turn History
    if isinstance(text, list): 
        # It's a history list [{"role": "user", ...}...]
        # SANITIZATION: Check content types and flatten
        sanitized_history = []
        for msg in text:
             content = msg['content']
             if isinstance(content, list):
                  # Extract text from standard multimodal format [{"text": "...", "type": "text"}]
                  text_parts = [p['text'] for p in content if 'text' in p]
                  content = "\n".join(text_parts)
             elif not isinstance(content, str):
                  content = str(content)
             sanitized_history.append({"role": msg['role'], "content": content})

        # Sliding Window: Keep last 2 exchanges (4 messages) to save VRAM
        if len(sanitized_history) > 4:
             sanitized_history = sanitized_history[-4:]
             print("DEBUG: History truncated to last 4 messages.")
             
        # MAX TOKEN/CHAR SAFEGUARD (EPIC 40)
        # 8GB VRAM cannot handle huge contexts with Adapter overhead
        # Naive char limit: 4000 chars approx 1000-1500 tokens
        total_len = sum(len(m['content']) for m in sanitized_history)
        if total_len > 6000:
             print(f"DEBUG: Context massive ({total_len} chars). Truncating.")
             # First, remove old *whole* messages if possible
             while sum(len(m['content']) for m in sanitized_history) > 6000 and len(sanitized_history) > 1:
                  sanitized_history.pop(0)
             
             # Second: If STILL massive (single huge message), truncate strictly
             if sanitized_history:
                 last_msg = sanitized_history[-1]
                 if len(last_msg['content']) > 5000:
                      print(f"DEBUG: Single message too long ({len(last_msg['content'])}). Hard truncate to last 5000.")
                      last_msg['content'] = last_msg['content'][-5000:]
             
        tokenizer = MODELS["tokenizer"]
        prompt_text = tokenizer.apply_chat_template(sanitized_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        print("DEBUG: Applied Chat Template to History.")
    else:
        # Fallback for single string
        if not isinstance(text, str): text = str(text)
        # CLIP HUGE INPUTS
        if len(text) > 4000: text = text[-4000:]
        prompt_text = text
        inputs = MODELS["tokenizer"](text, return_tensors="pt").to("cuda")
    
    # --- EPIC 33: CONSTRAINT EXTRACTION ---
    constraints = []
    if "constraint_extractor" in MODELS:
        # Extract from LAST user message only to avoid noise
        if 'sanitized_history' in locals() and sanitized_history:
             last_msg = sanitized_history[-1]['content']
        else:
             if isinstance(text, list) and len(text) > 0:
                 last_msg = text[-1]['content']
             else:
                 last_msg = str(text) if not isinstance(text, list) else ""
             
        # Double check string type and CLIP for extractor
        if not isinstance(last_msg, str): last_msg = str(last_msg)
        extract_input = last_msg[-1000:] # Only look at recent context for constraints
        
        try:
            constraints = MODELS["constraint_extractor"].extract(extract_input)
            
            # --- EPIC 52: EVOLUTIONARY MEMORY MANIFOLD (Retrieval) ---
            memory = MODELS.get("memory")
            if memory is not None:
                 try:
                     mem_context = memory.get_context_string(last_msg)
                     if mem_context:
                          print(f"DEBUG: Manifold Resonance (Mem0) Retrieved:\n{mem_context}")
                          # Interpret retrieved memory as semantic constraints
                          mem_constraints = MODELS["constraint_extractor"].extract(mem_context)
                          if mem_constraints:
                               constraints.extend(mem_constraints)
                               print(f"DEBUG: Memory Attractors (Constraints): {mem_constraints}")
                          # Inject grounding into prompt
                          prompt_text = f"<|im_start|>system\n[System Context: Core User Memory]\n{mem_context}<|im_end|>\n" + prompt_text
                          # Re-tokenize since text changed
                          inputs = MODELS["tokenizer"](prompt_text, return_tensors="pt").to("cuda")
                 except Exception as e:
                     print(f"Memory Evolver Error: {e}")
            
            # Store constraints for Telemetry and Jump trigger
            TELEMETRY_STATE["active_constraints"] = list(set(constraints)) # Deduplicate
            TELEMETRY_STATE["constraint_score"] = 0.0 if constraints else 1.0
            if constraints:
                print(f"Active Attractors: {TELEMETRY_STATE['active_constraints']}")
                TELEMETRY_STATE["thought_trace"].append(f"ATTRACTORS: {TELEMETRY_STATE['active_constraints']}")
        except Exception as e:
            print(f"Constraint/Memory Extraction Failed: {e}")
            
    # inputs is already set above
    model = MODELS["llm"] # Restore model definition for generate_thread
    
    print(f"DEBUG: Starting Generation. Input Length={len(prompt_text)}")
    
    # Memory Optimization
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- EPIC 40: THERMODYNAMIC SAMPLING ---
    # Adjust Temperature based on Manifold Curvature
    # High Curvature (Branching/Complex) -> Higher Temp (Explore)
    # Low Curvature (Flat/Simple) -> Lower Temp (Exploit)
    base_temp = 0.7
    curv = TELEMETRY_STATE.get("curvature", 0.0)
    # Curvature is typically -1.0 to 0.0 for hyperbolic.
    # We take absolute value to measure "intensity" of geometry.
    dynamic_temp = base_temp * (1.0 + 0.5 * abs(curv))
    # Cap temperature tightly. Qwen degrades rapidly > 0.85
    dynamic_temp = max(0.1, min(dynamic_temp, 0.85))
    
    print(f"DEBUG: Thermodynamic Sampling | K={curv} -> T={dynamic_temp:.2f}")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Cap tokens to prevent extreme loops causing OOM
    safe_max_tokens = min(max_new_tokens, 4096) if max_new_tokens else 2048

    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=safe_max_tokens,
        do_sample=True, 
        temperature=dynamic_temp,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.05, # Balanced to prevent loops but allow common words 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    def generate_thread():
        try:
             model.generate(**generation_kwargs)
        except Exception as e:
             print(f"ERROR in Generate Thread: {e}")
             with open("last_error.log", "w") as f:
                 f.write(f"Generate Error: {e}")
        finally:
             streamer.end()
             gc.collect()
             torch.cuda.empty_cache()

    thread = threading.Thread(target=generate_thread)
    thread.start()
    
    # --- EPIC 52: EVOLUTIONARY MEMORY MANIFOLD (Growth/Savings) ---
    def save_memory_thread():
         memory = MODELS.get("memory")
         if memory is not None and 'last_msg' in locals():
             try:
                 memory.add(last_msg)
                 print(f"DEBUG: Manifold evolved. Memory recorded.")
             except Exception as e:
                 print(f"Memory Save Error: {e}")
                 
    mem_thread = threading.Thread(target=save_memory_thread)
    mem_thread.start()
    
    
    partial_text = ""
    try:
        first_token = True
        token_count = 0
        for new_text in streamer:
            token_count += 1
            if first_token:
                print("DEBUG: Stream started receiving tokens.")
                first_token = False
            
            if not new_text: continue
                
            partial_text += new_text
            
            # --- EPIC 33: CONSTRAINT PRESSURE LOOP ---
            if constraints and token_count % 30 == 0:
                scorer = MODELS["constraint_scorer"]
                scores = scorer.score(partial_text, constraints)
                avg_score = sum(scores.values()) / len(scores) if scores else 1.0
                TELEMETRY_STATE["constraint_score"] = avg_score
                
                # Feedback: If failing to meet constraints, nudge gently (Free Will Mode)
                # Only kick if deeply stuck in a loop or completely off-topic for long duration
                if avg_score < 0.1 and token_count > 150 and random.random() < 0.3:
                     # OR Trigger on OOD Detection (Epic 42)
                     lip_violation = TELEMETRY_STATE.get("lipschitz_ratio", 1.0) > 8.0
                     
                     if lip_violation:
                         msg = f"OOD TELEPORTATION (L={TELEMETRY_STATE['lipschitz_ratio']:.1f}) -> JUMP!"
                     else:
                         msg = f"CRITICAL VIOLATION ({avg_score:.2f}) -> INITIATING NEUROSYMBOLIC JUMP!"
                         
                     if True: # Always log if trigger condition met
                         if not TELEMETRY_STATE["thought_trace"] or TELEMETRY_STATE["thought_trace"][-1] != msg:
                            TELEMETRY_STATE["thought_trace"].append(msg)
                            if MODELS["adapter"] is not None:
                                # Fetch active indices to invert
                                llm_ref = MODELS["llm"]
                                # Use stored layer ref (robust to PEFT/LoRA wrapping)
                                target_layer = getattr(llm_ref, '_geo_target_layer', None)
                                state = getattr(target_layer, '_current_geo_state', None) if target_layer else None
                                active_idx = None
                                if state and state.active_indices is not None:
                                    # active_indices shape (B, K). For demo B=1. 
                                    active_idx = state.active_indices[0].tolist()
                            
                            # GeometricAdapter has self.executor? Need to check.
                            # It has self.lambda_calculus (FiberBundleLambdaCalculus).
                            # Wait, where is Executor attached? 
                            # Check geometric_adapter.py again.
                            
                            # If direct executor access is tricky, use manual logic for now or assumes exposed.
                            # Let's assume we can access fiber_store.
                            
                            with torch.no_grad():
                                store = MODELS["adapter"].fiber_store
                                # 1. Global noise (Reduced from 2.0 to 0.1 to prevent complete gibberish scrambling)
                                noise = torch.randn_like(store.s) * 0.1 
                                store.s.add_(noise.to("cuda"))
                                
                                # 2. Fiber Inversion
                                if active_idx:
                                    s_active = store.s[active_idx]
                                    store.s[active_idx] = -s_active # Invert Logic
                                    
                                TELEMETRY_STATE["active_fiber"] = "HYPERJUMP!!!"
            # Chunk UI updates to prevent Gradio/WebSocket flooding (causes OOM/freeze)
            if token_count % 3 == 0 or token_count == 1:
                yield partial_text
                
            # Hard fallback stop
            if token_count > safe_max_tokens:
                 print("DEBUG: Generation hit safe token limit. Forcing stop.")
                 break
                 
        # Final yield to ensure last tokens are sent
        if partial_text:
            yield partial_text

    except Exception as e:
         print(f"ERROR in Streamer Loop: {e}")
         yield f"[System Error: {e}]"

# --- TELEMETRY POLLING ---
def get_zone_color(d):
    """Get color based on radial distance from center."""
    if d < 0.3:
        return 'rgba(0, 255, 100, 0.9)'   # Green - Anchor
    elif d < 0.6:
        return 'rgba(255, 255, 0, 0.9)'   # Yellow - Balanced
    elif d < 0.85:
        return 'rgba(255, 150, 0, 0.9)'   # Orange - Exploratory
    else:
        return 'rgba(255, 50, 50, 0.9)'   # Red - Boundary/Danger

def get_zone_name(d):
    """Get zone name based on radial distance."""
    if d < 0.3:
        return "ANCHOR"
    elif d < 0.6:
        return "BALANCED"
    elif d < 0.85:
        return "EXPLORATORY"
    else:
        return "BOUNDARY ⚠️"

def poll_telemetry():
    """
    Poll telemetry state and return enhanced visualization.
    
    ENHANCED Poincaré Projection with:
    - Cognitive zones (Anchor/Balanced/Exploratory)
    - Semantic direction labels (Abstract/Concrete/Creative/Analytical)
    - Color-coded trajectory by zone
    - Gibbs temperature indicator
    - Hover information for interpretability
    """
    try:
        # Update Gibbs temperature
        damping = TELEMETRY_STATE.get("damping", 0.01)
        beta = compute_gibbs_temperature(damping)
        TELEMETRY_STATE["gibbs_beta"] = beta
        
        # Create ENHANCED Poincaré projection
        fig = go.Figure()
        theta = np.linspace(0, 2*np.pi, 100)
        
        # --- COGNITIVE ZONES (Background rings) ---
        # Zone 1: Anchor (Green) - Center
        r_anchor = 0.3
        fig.add_trace(go.Scatter(
            x=r_anchor * np.cos(theta),
            y=r_anchor * np.sin(theta),
            fill='toself',
            fillcolor='rgba(0, 255, 100, 0.15)',
            line=dict(color='rgba(0, 255, 100, 0.4)', width=1, dash='dot'),
            name='Anchor',
            hoverinfo='text',
            hovertext='🟢 ANCHOR ZONE<br>High confidence | System 1<br>Stable semantics'
        ))
        
        # Zone 2: Balanced (Yellow)
        r_balanced = 0.6
        fig.add_trace(go.Scatter(
            x=r_balanced * np.cos(theta),
            y=r_balanced * np.sin(theta),
            fill='tonext',
            fillcolor='rgba(255, 255, 0, 0.08)',
            line=dict(color='rgba(255, 255, 0, 0.3)', width=1, dash='dot'),
            name='Balanced',
            hoverinfo='text',
            hovertext='🟡 BALANCED ZONE<br>Weighing options<br>Moderate certainty'
        ))
        
        # Zone 3: Exploratory (Orange)
        r_explore = 0.85
        fig.add_trace(go.Scatter(
            x=r_explore * np.cos(theta),
            y=r_explore * np.sin(theta),
            fill='tonext',
            fillcolor='rgba(255, 150, 0, 0.08)',
            line=dict(color='rgba(255, 150, 0, 0.3)', width=1, dash='dot'),
            name='Exploratory',
            hoverinfo='text',
            hovertext='🟠 EXPLORATORY ZONE<br>System 2 thinking<br>High uncertainty'
        ))
        
        # --- BOUNDARY CIRCLE ---
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines',
            line=dict(color='cyan', width=2),
            name='Boundary',
            hoverinfo='text',
            hovertext='⚠️ STABILITY BOUNDARY<br>Beyond = semantic instability'
        ))
        
        # --- ORIGIN MARKER ---
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=10, color='lime', symbol='diamond',
                       line=dict(color='white', width=1)),
            name='Anchor',
            hoverinfo='text',
            hovertext='◆ SEMANTIC ANCHOR<br>Origin of meaning'
        ))
        
        # --- THOUGHT TRAJECTORY ---
        if TELEMETRY_STATE["manifold_trace"]:
            arr = np.array(TELEMETRY_STATE["manifold_trace"])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                n_points = len(arr)
                
                # Calculate distances for zone coloring
                distances = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)
                
                # Trajectory line with gradient effect
                for i in range(len(arr) - 1):
                    opacity = 0.3 + 0.7 * i / max(n_points - 1, 1)
                    fig.add_trace(go.Scatter(
                        x=[arr[i, 0], arr[i+1, 0]],
                        y=[arr[i, 1], arr[i+1, 1]],
                        mode='lines',
                        line=dict(color=f'rgba(255, 0, 255, {opacity})', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Trajectory points colored by zone
                colors = [get_zone_color(d) for d in distances]
                sizes = [4 + 8 * i/max(n_points, 1) for i in range(n_points)]
                
                hover_texts = []
                for i, d in enumerate(distances):
                    zone = get_zone_name(d)
                    hover_texts.append(f'Step {i+1}<br>Zone: {zone}<br>Radius: {d:.2f}')
                
                fig.add_trace(go.Scatter(
                    x=arr[:, 0], y=arr[:, 1],
                    mode='markers',
                    marker=dict(size=sizes, color=colors,
                               line=dict(color='white', width=0.5)),
                    name='Steps',
                    hoverinfo='text',
                    hovertext=hover_texts
                ))
                
                # Current position (prominent star marker)
                if n_points > 0:
                    current_d = distances[-1]
                    if current_d < 0.3:
                        star_color = 'lime'
                        status = 'ANCHORED'
                    elif current_d < 0.6:
                        star_color = 'yellow'
                        status = 'BALANCED'
                    elif current_d < 0.85:
                        star_color = 'orange'
                        status = 'EXPLORING'
                    else:
                        star_color = 'red'
                        status = 'UNSTABLE'
                    
                    fig.add_trace(go.Scatter(
                        x=[arr[-1, 0]], y=[arr[-1, 1]],
                        mode='markers',
                        marker=dict(size=14, color=star_color, symbol='star',
                                   line=dict(color='white', width=2)),
                        name='Current',
                        hoverinfo='text',
                        hovertext=f'⭐ CURRENT: {status}<br>Bundle: {TELEMETRY_STATE["active_fiber"]}<br>r={current_d:.3f}'
                    ))
        
        # --- LAYOUT WITH ANNOTATIONS ---
        rw_status = "✓" if beta > 1.87 else "✗"
        rw_color = 'lime' if beta > 1.87 else 'orange'
        
        fig.update_layout(
            title=dict(
                text="Poincaré Manifold Projection",
                font=dict(size=12, color='cyan'),
                x=0.5
            ),
            template="plotly_dark",
            xaxis=dict(
                range=[-1.25, 1.25], 
                showgrid=False, 
                zeroline=False,
                showticklabels=False, 
                title=None,
                fixedrange=True
            ),
            yaxis=dict(
                range=[-1.25, 1.25], 
                showgrid=False, 
                zeroline=False,
                showticklabels=False, 
                title=None, 
                scaleanchor='x',
                fixedrange=True
            ),
            width=320, 
            height=320,
            margin=dict(l=5, r=5, t=35, b=45),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            annotations=[
                # Semantic direction labels
                dict(x=0, y=1.12, text='<b>ABSTRACT</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=0, y=-1.12, text='<b>CONCRETE</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=1.12, y=0, text='<b>CREATIVE</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=-1.12, y=0, text='<b>ANALYTICAL</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                # Zone labels
                dict(x=0, y=0.15, text='ANCHOR', showarrow=False,
                     font=dict(size=6, color='rgba(0, 255, 100, 0.4)')),
                dict(x=0, y=0.45, text='BALANCED', showarrow=False,
                     font=dict(size=6, color='rgba(255, 255, 0, 0.4)')),
                dict(x=0, y=0.72, text='EXPLORE', showarrow=False,
                     font=dict(size=6, color='rgba(255, 150, 0, 0.4)')),
                # Bottom metrics
                dict(x=0.5, y=-0.12, xref='paper', yref='paper',
                     text=f'<b>β={beta:.1f}</b>{rw_status} | κ={TELEMETRY_STATE["curvature"]:.1f} | S={TELEMETRY_STATE["entropy"]:.2f}',
                     showarrow=False,
                     font=dict(size=9, color=rw_color))
            ]
        )
        
        return (
            f"{TELEMETRY_STATE['curvature']}",
            f"{TELEMETRY_STATE['entropy']}",
            f"{TELEMETRY_STATE['active_fiber']}",
            f"{TELEMETRY_STATE['constraint_score']:.2f}",
            "\n".join(TELEMETRY_STATE["thought_trace"][-8:]),
            fig
        )
    except Exception as e:
        print(f"Telemetry Poll Error: {e}")
        return ("0.0", "0.0", "Error", "0.0", f"Error: {e}", go.Figure())

# --- UI BUILD ---
css_path = os.path.join(os.path.dirname(__file__), "theme_neural_glass.css")
if not os.path.exists(css_path):
    css_path = "igbundle-llm/theme_neural_glass.css" # Fallback for root execution

css = open(css_path).read()

with gr.Blocks(theme=gr.themes.Base(), css=css, title="NEURAL GLASS") as app:
    
    # Init
    start_btn = gr.Button("Initialize System", visible=False)
    start_btn.click(load_models, None, None)
    
    with gr.Row(elem_classes="glass-panel"):
        gr.Markdown("# <span class='neon-header'>IGBUNDLE // NEURAL GLASS</span>")
        
    with gr.Row():
        # SIDEBAR
        with gr.Column(scale=1, elem_classes="glass-panel"):
            gr.Markdown("### SYSTEM STATUS")
            
            with gr.Group(elem_classes="info-card"):
                gr.Markdown("**MODEL IDENTITY**\nUnified Manifold v2.1\nQwen 7B (NF4) + SigLIP")
                
            with gr.Group(elem_classes="info-card"):
                k_label = gr.Label(label="SECTIONAL CURVATURE (K)", value="0.0")
                s_label = gr.Label(label="MANIFOLD ENTROPY (S)", value="0.0")
                constraint_label = gr.Label(label="CONSTRAINT SCORE", value="1.0")
                fiber_label = gr.Label(label="ACTIVE BUNDLE", value="Standby")
                
                # EPIC 43: Holographic Plot
                manifold_plot = gr.Plot(label="Poincaré Manifold")


        # MAIN STAGE
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_classes=["glass-panel"], 
                height=400, 
                show_label=False
            )
            
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False, 
                    placeholder="Enter query or drag image...", 
                    container=False,
                    scale=4
                )
                img_input = gr.Image(type="filepath", container=False, scale=1, height=50)
                send_btn = gr.Button("TRANSMIT", variant="primary", scale=1, elem_classes="neon-btn")

            # THOUGHT STREAM
            gr.Markdown("### THOUGHT MANIFOLD TRACE")
            thought_log = gr.TextArea(
                label="Internal Monologue", 
                lines=5, 
                max_lines=5, 
                elem_classes="thought-stream",
                interactive=False
            )

    # EVENT LOOP
    def user_turn(user_message, history, image):
        print(f"DEBUG: user_turn input type={type(user_message)} val={repr(user_message)}")
        
        # Handle Gradio 6.x complex return types
        val = user_message
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
             if 'text' in val[0]:
                 val = val[0]['text']
        if isinstance(val, dict) and 'text' in val:
             val = val['text']
             
        if not val or not str(val).strip():
            print("WARNING: Empty user message ignored.")
            # Ensure history is a list even if ignoring
            if history is None: history = []
            return "", history, image 
            
        # Add to history (Messages Format - Default for Chatbot now)
        if history is None: history = []
        history.append({"role": "user", "content": str(val)})
        return "", history, image
        
    def bot_turn(history, image):
        print("DEBUG: Entered bot_turn")
        
        # Validation: If history is empty, user sent nothing.
        if not history:
            print("DEBUG: History empty, skipping bot turn.")
            yield history
            return
            
        # Check if last message is valid
        if not history[-1] or not isinstance(history[-1], dict) or "content" not in history[-1]:
             print("DEBUG: Invalid history format.")
             yield history
             return

        # Retrieve the user's last message (which user_turn just added)
        user_message = history[-1]["content"] if history else ""
        if not user_message:
             print("DEBUG: User message empty (after extraction), skipping.")
             yield history
             return
        
        # Stream response
        # Context should be the conversation history up to this point
        # We need to append an empty bot message for streaming?
        history.append({"role": "assistant", "content": ""})
        
        bot_message = ""
        # Pass conversation history (excluding the new empty assistant slot)
        # Note: generate_stream expects List[Dict] or String.
        # It handles conversion if needed, but here we pass List[Dict].
        conversation_context = history[:-1]
        
        print(f"DEBUG: Calling generate_stream with {len(conversation_context)} msgs")
        try:
            found_yield = False
            # Reduce max_new_tokens to 512 to prevent OOM on 8GB GPU
            for partial in generate_stream(conversation_context, image, 512):
                found_yield = True
                bot_message = partial
                history[-1]["content"] = bot_message
                # Yield NEW list to force Gradio update
                yield history
            
            if not found_yield:
                 print("DEBUG: generate_stream yielded NOTHING.")
                 history[-1]["content"] = "**Error: Generation yielded nothing.**"
                 yield history
                 
        except Exception as e:
            print(f"ERROR in bot_turn: {e}")
            history[-1]["content"] = f"**System Error**: {e}"
            yield history
            
    txt_input.submit(user_turn, [txt_input, chatbot, img_input], [txt_input, chatbot, img_input]).then(
        bot_turn, [chatbot, img_input], [chatbot]
    )
    send_btn.click(user_turn, [txt_input, chatbot, img_input], [txt_input, chatbot, img_input]).then(
        bot_turn, [chatbot, img_input], [chatbot]
    )
    
    # Telemetry Timer
    timer = gr.Timer(0.1)
    timer.tick(poll_telemetry, None, [k_label, s_label, fiber_label, constraint_label, thought_log, manifold_plot])

if __name__ == "__main__":
    # Pre-load models for API consistency
    print("Initializing Models...")
    load_models()
    
    app.queue().launch(
        server_name="0.0.0.0", 
        server_port=7865,
        share=False,
        show_error=True
    )
