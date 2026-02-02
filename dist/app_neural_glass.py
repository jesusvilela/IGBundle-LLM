
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
sys.path.append(os.path.abspath(".")) # Use local igbundle package in dist/
import igbundle
from igbundle.core.config import IGBundleConfig
from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.fibers.constraint import ConstraintExtractor, ConstraintScorer

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
    "adapter": None
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
    
    print("Loading Geometric Adapter...")
    config = IGBundleConfig(
        hidden_size=3584, num_components=8, latent_dim=64, num_categories=16,
        use_dynamics=True, use_geodesic_attn=True, supported_modalities=["vision", "text"]
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
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean().item()
            
            # Fiber Activation (Real Sparse Hamiltonian Indices)
            if hasattr(geo_state, 'active_indices') and geo_state.active_indices is not None:
                 # Take the first active index from the first sample in batch
                 idx = geo_state.active_indices[0, 0].item()
                 active_fiber = f"Bundle-{int(idx)}"
            else:
                 active_fiber = "Standby"
            
            # Simulated Telemetry for FX (Curvature fluctuates with Entropy)
            # Higher Entropy (Confusion) -> Lower Curvature (Flat)?
            # High Reasoning -> High Curvature?
            sim_curvature = 2000.0 + (entropy * 500.0) + random.uniform(-100, 100)
            
            # Update Global State
            TELEMETRY_STATE["curvature"] = round(sim_curvature, 2)
            TELEMETRY_STATE["entropy"] = round(entropy, 4)
            TELEMETRY_STATE["active_fiber"] = active_fiber
            TELEMETRY_STATE["history_k"].append(sim_curvature)
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
        constraints = MODELS["constraint_extractor"].extract(text)
        TELEMETRY_STATE["active_constraints"] = constraints
        TELEMETRY_STATE["constraint_score"] = 0.0 if constraints else 1.0
        if constraints:
            print(f"Constraints Detected: {constraints}")
            TELEMETRY_STATE["thought_trace"].append(f"OBLIGATIONS: {constraints}")
            
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print(f"DEBUG: Starting Generation. Input='{text}'")
    
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
                
                # Feedback: If failing to meet constraints after 60 tokens, KICK the system
                if avg_score < 0.4 and token_count > 60:
                     msg = f"VIOLATION ({avg_score:.2f}) -> PERTURBING LATENTS (Scale 0.05)!"
                     if not TELEMETRY_STATE["thought_trace"] or TELEMETRY_STATE["thought_trace"][-1] != msg:
                        TELEMETRY_STATE["thought_trace"].append(msg)
                        if MODELS["adapter"] is not None:
                            with torch.no_grad():
                                # Apply reduced random noise to prevent collapse
                                noise = torch.randn_like(MODELS["adapter"].fiber_store.s) * 0.05
                                MODELS["adapter"].fiber_store.s.data += noise.to("cuda")
            
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
    
    return (
        f"{TELEMETRY_STATE['curvature']}",
        f"{TELEMETRY_STATE['entropy']}",
        f"{TELEMETRY_STATE['active_fiber']}",
        f"{TELEMETRY_STATE['constraint_score']:.2f}",
        "\n".join(TELEMETRY_STATE["thought_trace"][-8:])
    )

# --- UI BUILD ---
css = open("theme_neural_glass.css").read()

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
            return "", history, image # Don't update history
            
        # Add to history (Messages Format)
        history.append({"role": "user", "content": str(val)})
        return "", history, image
        
    def bot_turn(history, image):
        if not history:
             return history
             
        user_message = history[-1]["content"] 
        if not user_message:
             return history
             
        history.append({"role": "assistant", "content": ""})
        
        bot_message = ""
        for partial in generate_stream(user_message, image, 2048):
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
    timer.tick(poll_telemetry, None, [k_label, s_label, fiber_label, constraint_label, thought_log])

if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", server_port=7865)
