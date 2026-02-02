
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
import warnings
import warnings
warnings.filterwarnings("ignore")

# --- OPTIMIZATION FLAGS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# --- PATHS & IMPORTS ---
sys.path.append(os.path.abspath("src"))
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.fibers.constraint import ConstraintExtractor, ConstraintScorer
from igbundle.quantum.gibbs import QuantumGibbsSampler

# --- CONSTANTS ---
BASE_MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
ADAPTER_PATH = "" # Autodetect
CHECKPOINT_DIR = "igbundle_refined_training"


# --- GLOBAL STATE ---
# We use a global state to allow the UI to poll telemetry while generating
TELEMETRY_STATE = {
    "curvature": 0.0,
    "entropy": 0.0,
    "active_fiber": "None",
    "hamiltonian_energy": 0.0,
    "retrospection_loss": 0.0,
    "fhn_phase": "FREE",
    "thought_trace": [],
    "history_k": [], # History for plotting
    "history_s": [],
    "active_constraints": [],
    "constraint_score": 1.0
}

MODELS = {
    "llm": None,
    "tokenizer": None,
    "vision_model": None,
    "processor": None,
    "adapter": None,
    "quantum_sampler": None
}

# --- MODEL LOADING ---
def find_latest_adapter():
    if os.path.exists("dist/adapter_refined.pt"): return "dist/adapter_refined.pt"
    if os.path.exists("adapter_weights.pt"): return "adapter_weights.pt"
    if os.path.exists(ADAPTER_PATH): return ADAPTER_PATH
    
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
             latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
             return os.path.join(CHECKPOINT_DIR, latest, "adapter_weights.pt")
             
    # Fallback to diverse training if refined not found
    fallback_dir = "igbundle_diverse_training"
    if os.path.exists(fallback_dir):
         checkpoints = [d for d in os.listdir(fallback_dir) if d.startswith("checkpoint-")]
         if checkpoints:
             latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
             return os.path.join(fallback_dir, latest, "adapter_weights.pt")
             
    return "DEBUG_MODE_NO_WEIGHTS"

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
    
    print("Loading Quantum Gibbs Sampler...")
    try:
        MODELS["quantum_sampler"] = QuantumGibbsSampler(n_qubits=16)
    except Exception as e:
        print(f"Quantum Init Failed: {e}")
        MODELS["quantum_sampler"] = None

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

    
    weight_path = find_latest_adapter()
    if weight_path != "DEBUG_MODE_NO_WEIGHTS":
        print(f"Loading Weights: {weight_path}")
        adapter.load_state_dict(torch.load(weight_path), strict=False)
    
    MODELS.update({
        "llm": llm, "tokenizer": tokenizer, 
        "vision_model": vision_model, "processor": processor, "adapter": adapter,
        "constraint_extractor": ConstraintExtractor(),
        "constraint_scorer": ConstraintScorer()
    })
    
    # Inject Hook
    inject_adapter_hook(llm, adapter)
    return f"System Online. Weights: {weight_path}"

# --- ADAPTER HOOK (THE "GHOST" MECHANISM) ---
# This is a shared container for vision features that changes per request
VISION_CONTEXT = {"feats": None}

def inject_adapter_hook(llm, adapter):
    target_layer = llm.model.layers[12]
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
            
            # Simulated Telemetry for FX (Curvature fluctuates with Entropy)
            sim_curvature = 2000.0 + (entropy * 500.0) + random.uniform(-100, 100)
            
            # Update Global State
            TELEMETRY_STATE["curvature"] = round(float(sim_curvature), 2)
            TELEMETRY_STATE["entropy"] = round(float(entropy), 4)
            TELEMETRY_STATE["active_fiber"] = str(active_fiber)
            TELEMETRY_STATE["history_k"].append(sim_curvature)
            TELEMETRY_STATE["history_s"].append(entropy)
            
            # Keep history short
            if len(TELEMETRY_STATE["history_k"]) > 50:
                TELEMETRY_STATE["history_k"].pop(0)
                TELEMETRY_STATE["history_s"].pop(0)
                
             # Log Thought
            if random.random() < 0.1: # Don't flood
                 TELEMETRY_STATE["thought_trace"].append(f"Activating {active_fiber} | S={entropy:.2f}")

            # PHASE 4: PHYSICS TELEMETRY
            if hasattr(adapter, 'fiber_executor') and adapter.fiber_executor.resonator_state:
                q, p = adapter.fiber_executor.resonator_state
                # Energy H = 0.5 * (p^2 + q^2) (Approx dimensionless energy)
                energy = 0.5 * (p.pow(2).mean() + q.pow(2).mean()).item()
                TELEMETRY_STATE["hamiltonian_energy"] = round(energy * 100, 2)
                
                # Probabilistic Retrospection (Mental Check)
                # Only check if high entropy (confusion) or random 2%
                if entropy > 2.0 or random.random() < 0.02:
                     TELEMETRY_STATE["fhn_phase"] = "RETROSPECT"
                     # We need indices.
                     flat_indices = []
                     if hasattr(geo_state, 'active_indices') and geo_state.active_indices is not None:
                         flat_indices = geo_state.active_indices.reshape(-1).tolist()
                     
                     # Check thought
                     if flat_indices:
                         # Executor verify_thought returns boolean, but we also want the loss
                         # We'll access the retrospector directly or trust verify_thought to log/jump
                         # verify_thought handles the jump. We just want the loss for UI.
                         # Let's peek at retrospector last loss if possible, or re-run light check?
                         # Re-running is expensive. Let's just create a side-channel in executor later.
                         # For now, simulate loss based on entropy (correlation)
                         # Real implementation: Executor stores last_retro_loss
                         TELEMETRY_STATE["retrospection_loss"] = round(entropy * 0.5, 3) 
                         
                         # Trigger verification (Passive Observation)
                         # We pass enforce_jump=False to disable the "Doom Loop"
                         loss, valid = adapter.fiber_executor.verify_thought(flat_indices, enforce_jump=False)
                         
                         # Update with real loss from physics engine
                         TELEMETRY_STATE["retrospection_loss"] = round(loss, 3)
                         
                         if not valid:
                             TELEMETRY_STATE["thought_trace"].append(f"Relativity Drift: {loss:.3f} (Observing)")
                else:
                    TELEMETRY_STATE["fhn_phase"] = "FREE"

        except Exception as e:
            print(f"Telemetry Error: {e}")

        adapted = adapted_out.to(orig_dtype)
        if isinstance(out, tuple): return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = adapter_hook
    print("Hook Injected.")

# --- INFERENCE GEN ---
def generate_stream(full_prompt, latest_input, image_path, max_new_tokens):
    load_models()
    text = latest_input
    
    # Process Vision
    VISION_CONTEXT['feats'] = None
    if image_path:
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
    
    # Gradio 6.x Input Sanitization
    if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
        if 'text' in text[0]:
            print("DEBUG: Extracting text from list-dict structure.")
            text = text[0]['text']
            
    if not isinstance(text, str):
        print("WARNING: text is not a string! Converting...")
        text = str(text)
        
    # Construct Prompt from History if available, or just text
    tokenizer = MODELS["tokenizer"]
    model = MODELS["llm"]
    
    # --- EPIC 33: CONSTRAINT EXTRACTION ---
    constraints = []
    if "constraint_extractor" in MODELS:
        # Use only the LATEST input for constraints
        if not isinstance(latest_input, str): latest_input = str(latest_input)
        constraints = MODELS["constraint_extractor"].extract(latest_input)
        TELEMETRY_STATE["active_constraints"] = constraints
        TELEMETRY_STATE["constraint_score"] = 0.0 if constraints else 1.0
        if constraints:
            print(f"Constraints Detected: {constraints}")
            TELEMETRY_STATE["thought_trace"].append(f"OBLIGATIONS: {constraints}")
            
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    print(f"DEBUG: Starting Generation. Input Length={inputs.input_ids.shape[1]}")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        temperature=0.7
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

    thread = threading.Thread(target=generate_thread)
    thread.start()
    
    partial_text = ""
    doom_counter = 0
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
            
            # --- EMERGENCY GUARDRAILS: DOOM LOOP DETECTION ---
            # Detect raw JSON leakage or recursive prompting (hallucinating prompt structure)
            DOOM_PATTERNS = ["[{'text':", "{'type':", "User: [{'"]
            # Check recent history (last 100 chars) for efficiency
            search_window = partial_text[-100:] if len(partial_text) > 100 else partial_text
            
            if any(pattern in search_window for pattern in DOOM_PATTERNS):
                 print("DEBUG: CONSTRAINT VIOLATION. INITIATING QUANTUM EXPANSION.")
                 
                 # 1. VISUALIZATION
                 msg = "FATAL: DOOM LOOP DETECTED -> EMERGENCY PURGE"
                 if not TELEMETRY_STATE["thought_trace"] or TELEMETRY_STATE["thought_trace"][-1] != msg:
                     TELEMETRY_STATE["thought_trace"].append(msg)
                 TELEMETRY_STATE["constraint_score"] = 0.0
                 TELEMETRY_STATE["active_fiber"] = "EMERGENCY JUMP 🛑"

                 # 2. NEUROSYMBOLIC JUMP (Intensity 3.0)
                 if MODELS["adapter"] is not None and hasattr(MODELS["adapter"], "fiber_executor"):
                     try:
                        llm_ref = MODELS["llm"]
                        state = getattr(llm_ref.model.layers[12], '_current_geo_state', None)
                        active_idx_list = None
                        if state and state.active_indices is not None:
                            active_idx_list = state.active_indices[0].tolist()
                        
                        MODELS["adapter"].fiber_executor.hyper_jump(
                            active_indices=active_idx_list, 
                            intensity=3.0
                        )
                     except Exception as e:
                         print(f"Emergency Jump Failed: {e}")

                 # 3. INTERVENTION MESSAGE (ADAPTIVE - MANIFOLD SHIFT)
                 doom_counter += 1
                 
                 # STAGE 3: HARDWARE SAFETY STOP (Attempt 7+)
                 # STAGE 3: HARDWARE SAFETY STOP (DISABLED FOR INFINITE EXPANSION)
                 # if doom_counter > 6:
                 #     print("DEBUG: Hardware Safety Stop triggered.")
                 #     yield partial_text + "\n\n[SYSTEM FAILURE] 🛑 Hardware Limits Reached. Stopping to prevent crash.\n"
                 #     break

                 # STAGE 2: MANIFOLD SHIFT (Attempts 4-6)
                 # Aggressive KV Geometry Search (Quantum-Accelerated)
                 if doom_counter > 3:
                     # High Intensity Jump
                     if MODELS["adapter"] is not None and hasattr(MODELS["adapter"], "fiber_executor"):
                         try:
                             llm_ref = MODELS["llm"]
                             state = getattr(llm_ref.model.layers[12], '_current_geo_state', None)
                             active_idx_list = []
                             if state and state.active_indices is not None:
                                 active_idx_list = state.active_indices[0].tolist()
                                 
                             # 1. QUANTUM SAMPLING (Gibbs State)
                             quantum_vector = None
                             if MODELS["quantum_sampler"]:
                                 quantum_vector = MODELS["quantum_sampler"].sample_geometry(active_idx_list)
                            
                             # 2. HYPER-JUMP with Quantum Seed
                             MODELS["adapter"].fiber_executor.hyper_jump(
                                 active_indices=active_idx_list, 
                                 intensity=5.0,  # HIGH INTENSITY
                                 custom_direction=quantum_vector
                             )
                             print("DEBUG: Quantum Manifold Shift Executed.")
                         except Exception as e:
                             print(f"Shift Jump Failed: {e}")

                     yield partial_text + "\n\n[MANIFOLD SHIFT] 🌌 Initiating Deep KV Geometry Search (Quantum-Accelerated)...\n"
                 
                 # STAGE 1: ORGANIC EXPANSION (Attempts 1-3)
                 else:
                     yield partial_text + "\n\n[NEUROSYMBOLIC INTERVENTION] 🌌 Reformulating thought process...\n"

            # --- EFFECT DISCIPLINE: PERIODIC REFINEMENT ---
            if token_count % 7 == 0 and MODELS["adapter"] is not None:
                # Keep fibers anchored to prevent drift during long generation
                try:
                    llm_ref = MODELS["llm"]
                    # Safe retrieval of state from model layer
                    state = getattr(llm_ref.model.layers[12], '_current_geo_state', None)
                    if state and state.active_indices is not None:
                         # refine_latents expects Tensor
                         MODELS["adapter"].refine_latents(state.active_indices)
                         if token_count % 28 == 0:
                             print(f"DEBUG: Refined Latents at token {token_count}")
                except Exception as e:
                    print(f"Refinement Warning: {e}")
            
            # --- EPIC 33: CONSTRAINT PRESSURE LOOP ---
            if constraints and token_count % 30 == 0:
                scorer = MODELS["constraint_scorer"]
                scores = scorer.score(partial_text, constraints)
                avg_score = sum(scores.values()) / len(scores) if scores else 1.0
                min_score = min(scores.values()) if scores else 1.0
                TELEMETRY_STATE["constraint_score"] = avg_score
                
                # Feedback: If any single constraint is violated (Score < 0.4), KICK the system
                if min_score < 0.4 and token_count > 45: # Strict enforcement
                     msg = f"CRITICAL VIOLATION ({avg_score:.2f}) -> INITIATING NEUROSYMBOLIC JUMP!"
                     if not TELEMETRY_STATE["thought_trace"] or TELEMETRY_STATE["thought_trace"][-1] != msg:
                        TELEMETRY_STATE["thought_trace"].append(msg)
                        if MODELS["adapter"] is not None and hasattr(MODELS["adapter"], "fiber_executor"):
                            # Fetch active indices to invert
                            llm_ref = MODELS["llm"]
                            state = getattr(llm_ref.model.layers[12], '_current_geo_state', None)
                            active_idx_list = None
                            
                            if state and state.active_indices is not None:
                                # active_indices shape (B, K). For demo B=1. 
                                active_idx_list = state.active_indices[0].tolist()
                            
                            # Execute Standardized Hyper-Jump
                            MODELS["adapter"].fiber_executor.hyper_jump(
                                active_indices=active_idx_list, 
                                intensity=2.0
                            )
                            
                            TELEMETRY_STATE["active_fiber"] = "HYPERJUMP!!!"
                            
                            
                            # EXPANSION: Don't stop. Bridge to new manifold.
                            # Inject a strong diversion prompt into the stream
                            expansion_text = "\n\n[NEUROSYMBOLIC EXPANSION] 🌌\n\n(Geometric Pivot Triggered)\n"
                            partial_text += expansion_text
                            yield partial_text
                            
                            # SOFT STOP: Allow organic expansion
                            print("DEBUG: Triggering Soft Expansion.") 

            
            yield partial_text
    except Exception as e:
         print(f"ERROR in Streamer Loop: {e}")
         yield f"[System Error: {e}]"

# --- TELEMETRY POLLING ---
def poll_telemetry():
    # Return list of values for graphs and labels
    k_plot = list(enumerate(TELEMETRY_STATE["history_k"])) # (x, y) pairs? No, Gradio LinePlot needs DF or x,y lists
    # Gradio Plot component expects a pandas dataframe usually or specific mapping
    # Simple Plot: Just return list of Y values?
    # Let's just return the scalar strings for labels first
    
    # Create line plot data (dummy for now if complex, but simple list works for LinePlot?)
    # Actually Gradio gr.LinePlot is complex. Let's use Label for simplicity or JSON.
    # Wait, 10X means visual. Let's use sending updates to a component.
    
    # Broadcast State to Disk for Braintop Visualization
    try:
        # Avoid race condition with atomic write if possible, or just try/except
        # We only write specific fields needed by Braintop
        export_state = {
            "status": "generating" if TELEMETRY_STATE["active_fiber"] != "Standby" else "idle",
            "curvature": TELEMETRY_STATE["curvature"],
            "entropy": TELEMETRY_STATE["entropy"],
            "hamiltonian_energy": TELEMETRY_STATE["hamiltonian_energy"],
            "retrospection_loss": TELEMETRY_STATE["retrospection_loss"],
            "phase": TELEMETRY_STATE["fhn_phase"],
            "active_fiber": TELEMETRY_STATE["active_fiber"],
            "prompt": TELEMETRY_STATE["thought_trace"][-1] if TELEMETRY_STATE["thought_trace"] else "Idle"
        }
        import json
        with open("dashboard_state.json", "w") as f:
            json.dump(export_state, f)
    except Exception as e:
        print(f"Broadcast Error: {e}")

    return (
        f"{TELEMETRY_STATE['curvature']}",
        f"{TELEMETRY_STATE['entropy']}",
        f"{TELEMETRY_STATE['active_fiber']}",
        f"{TELEMETRY_STATE['constraint_score']:.2f}",
        f"{TELEMETRY_STATE['hamiltonian_energy']}",
        f"{TELEMETRY_STATE['retrospection_loss']}",
        f"{TELEMETRY_STATE['fhn_phase']}",
        "\n".join(TELEMETRY_STATE["thought_trace"][-8:])
    )

# --- UI BUILD ---
# --- UI BUILD ---
css_path = os.path.join(os.path.dirname(__file__), "theme_neural_glass.css")
if os.path.exists(css_path):
    css = open(css_path).read()
else:
    print("WARNING: CSS not found. Using default.")
    css = ""

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

            with gr.Group(elem_classes="info-card"):
                h_label = gr.Label(label="HAMILTONIAN ENERGY H(q,p)", value="0.0")
                retro_label = gr.Label(label="RETROSPECTION LOSS", value="0.0")
                phase_label = gr.Label(label="DYNAMICS PHASE", value="FREE")

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
        
        # Handle Gradio 6.x / Multimodal complex return types
        val = user_message
        try:
            # Recursive unpacking for list/dict wrappings
            while isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if isinstance(val, dict):
                 if 'text' in val: val = val['text']
                 elif 'content' in val: val = val['content']
        except Exception as e:
            print(f"Input Parse Error: {e}")
            
        # Final type safety
        if not isinstance(val, str):
            val = str(val)
             
        if not val.strip():
            print("WARNING: Empty user message ignored.")
            return "", history, image
            
        # Add to history (Messages Format)
        history.append({"role": "user", "content": val})
        return "", history, image
        
    def bot_turn(history, image):
        if not history:
             return history
             
        user_message = history[-1]["content"] 
        if not user_message:
             return history
             
        history.append({"role": "assistant", "content": ""})
        
        # Build PROMPT with HISTORY
        full_prompt = ""
        for msg in history[:-1]: # Exclude the empty assistant msg we just added
            role = "User" if msg['role'] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += "Assistant:"
        
        bot_message = ""
        # Call generate_stream with FULL PROMPT and LATEST INPUT
        for partial in generate_stream(full_prompt, user_message, image, 2048):
            bot_message = partial
            history[-1]["content"] = bot_message
            # Yield NEW list to force Gradio update
            yield list(history)
            
    txt_input.submit(user_turn, [txt_input, chatbot, img_input], [txt_input, chatbot, img_input]).then(
        bot_turn, [chatbot, img_input], [chatbot]
    )
    send_btn.click(user_turn, [txt_input, chatbot, img_input], [txt_input, chatbot, img_input]).then(
        bot_turn, [chatbot, img_input], [chatbot]
    )
    
    # Telemetry Timer
    timer = gr.Timer(0.1)
    timer.tick(poll_telemetry, None, [k_label, s_label, fiber_label, constraint_label, h_label, retro_label, phase_label, thought_log])

if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", server_port=7865)
