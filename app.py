try:
    from igbundle.utils import triton_fix
except ImportError:
    # If package not installed in editable mode, try relative or local path
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from igbundle.utils import triton_fix

import os
from unsloth import FastLanguageModel
import argparse
import yaml
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from generate_braintop_viz import generate_viz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image as PILImage

# Global Viz State
VIZ_STATE = {
    "curvature": [],
    "affinity": [] # [Layers * Tokens, P, K]
}

class HookManager:
    """Manages forward hooks to capture intermediate geometric states."""
    def __init__(self, model):
        self.hooks = []
        self.model = model
        
    def _curvature_hook(self, module, input, output):
        # IGBundleAdapter output is (hidden_states, state)
        # state is a MixtureState object
        if isinstance(output, tuple) and len(output) > 1:
            state = output[1]
            # Capture base curvature (sigma)
            if hasattr(state, "sigma"):
                 VIZ_STATE["curvature"].append(state.sigma.detach().cpu().numpy())
            # Capture fiber affinity (p)
            if hasattr(state, "p"):
                 VIZ_STATE["affinity"].append(state.p.detach().cpu().numpy())

    def _affinity_hook(self, module, input, output):
        # Attention weights are tricky to hook in PEFT.
        # We might need to rely on the model returning attentions.
        pass

    def attach(self):
        # Find all IGBundle modules (Adapters)
        from igbundle.modules.adapter import IGBundleAdapter
        for name, module in self.model.named_modules():
            if isinstance(module, IGBundleAdapter):
                h = module.register_forward_hook(self._curvature_hook)
                self.hooks.append(h)
                print(f"Hooked adapter at {name} for curvature.")

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
def plot_curvature():
    """Generate a plot of the curvature distribution."""
    if not VIZ_STATE["curvature"]:
        return None
    
    # Sigmas are (B, T, P, D_lat). We take mean over D_lat for visualization.
    data = np.concatenate([c.mean(axis=-1).flatten() for c in VIZ_STATE["curvature"]])
    
    plt.figure(figsize=(8, 4))
    sns.histplot(data, bins=30, kde=True, color="purple")
    plt.title("Curvature $\sigma(x)$ Distribution (Riemannian Dispersion)")
    plt.xlabel("Local Curvature Value")
    plt.ylabel("Frequency")
    plt.axvline(x=1.0, color='r', linestyle='--', label="Ideal (1.0)")
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return PILImage.open(buf)

def plot_affinity():
    """Generate a plot of Category Affinities (Fiber Activations)."""
    if not VIZ_STATE["affinity"]:
        return None
        
    # Affinities are (B, T, P, K).
    # We aggregate across layers and time to see the "Active Fiber Map".
    # Mean across tokens and layers.
    data = np.mean(np.concatenate([a for a in VIZ_STATE["affinity"]], axis=1), axis=(0,1)) # (P, K)
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(data, annot=False, cmap="viridis", cbar_kws={'label': 'Activation Prob'})
    plt.title("Fiber Activation Map: Component (P) vs Category (K)")
    plt.xlabel("Category index $k$")
    plt.ylabel("Bundle Component $p$")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return PILImage.open(buf)

def load_model(config_path, checkpoint_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    base_model_id = cfg['base_model_id']
    
    print(f"Loading Base: {base_model_id}")
    if torch.cuda.is_available():
        free_gpu_mem, total_gpu_mem = torch.cuda.mem_get_info()
        free_gb = free_gpu_mem / 1024**3
        print(f"Free VRAM: {free_gb:.2f} GB / {total_gpu_mem / 1024**3:.2f} GB")
        if free_gb < 6.0:
            print("WARNING: Low VRAM detected. If training or other GPU apps are running, this may fail.")
            

    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Loading Model via Unsloth (4-bit Mode)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_path if checkpoint_path else base_model_id,
            max_seq_length = 1024,
            load_in_4bit = True,
            trust_remote_code = True,
            device_map = {"": 0}
        )
        FastLanguageModel.for_inference(model)
    else:
        print("CUDA not available. Loading in float32 on CPU via Transformers.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            trust_remote_code=True
        )

    
    # Inject IGBundle Adapter
    print("Injecting IGBundle Adapter...")
    if hasattr(model.config, "hidden_size"):
        print(f"Detected model hidden_size: {model.config.hidden_size}")
        cfg['ig_adapter']['hidden_size'] = model.config.hidden_size
        
    class DictConfig:
        def __init__(self, d):
            for k,v in d.items(): setattr(self, k, v)
    adapter_cfg = DictConfig(cfg['ig_adapter'])
    model = wrap_hf_candidate(model, adapter_cfg)
    
    # Reload weights if needed (FastLanguageModel might not load custom ig_adapter keys automatically)
    adapter_w_path = os.path.join(checkpoint_path, "adapter_weights.pt")
    if os.path.exists(adapter_w_path):
        print(f"Loading IGBundle explicit weights from {adapter_w_path}")
        model.load_state_dict(torch.load(adapter_w_path, map_location=model.device), strict=False)
    
    model.eval()
    return model, tokenizer

# Global Model
MODEL = None
TOKENIZER = None

def generate_response(message, history):
    if MODEL is None:
        return "Model not loaded."
        
    # Standard prompt for Instruction Tuning
    # Note: We can implement a system prompt logic here if desired, 
    # but for ChatInterface default we use (message, history).
    
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{message}\n\n### Response:\n"
    
    # Or use tokenizer.apply_chat_template if supported and we used it for training
    # For this demo, we trained on Alpaca text format manually constructed.
    
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=TOKENIZER.eos_token_id
        )
        
    # Decode
    generated = TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated

def generate_with_viz(message, history):
    if MODEL is None:
        return "Model not loaded.", None
        
    # Clear previous state
    VIZ_STATE["curvature"] = []
    VIZ_STATE["affinity"] = []
    
    # Generate
    response = generate_response(message, history)
    
    # Create Plots
    curv_plot = plot_curvature()
    aff_plot = plot_affinity()
    
    return response, curv_plot, aff_plot

def refresh_topology_view(checkpoint_path):
    """Generates a dynamic Braintop visualization based on current VIZ_STATE."""
    if not VIZ_STATE["affinity"]:
        # Fallback to base view if no dialogue yet
        output_file = "output/igbundle_topology_dynamic.html"
        generate_viz(checkpoint_path, output_file, lite_mode=True)
        return f'<iframe src="file/{output_file}" width="100%" height="800px"></iframe>'
        
    # Aggregate affinities across layers/tokens
    # affinity is a list of [B, T, P, K]
    # We want to map P (Components) or K (Categories) to nodes?
    # In generate_viz, we have num_nodes = D_bot (embeddings) or logic.
    # Actually, the adapter has P components and each has K categories.
    # Braintop nodes currently represent the "basis" of the hidden space.
    # Let's map THE MEAN ACTIVATION across layers to the nodes.
    
    # Simple heuristic: we map the first N activations to nodes
    aff_data = np.mean(np.concatenate([a for a in VIZ_STATE["affinity"]], axis=1), axis=(0,1)) # (P, K)
    
    # Flatten or select? Let's use the Component activations if we can map them.
    # For now, we'll map the top K categories to the first K nodes.
    node_activations = aff_data.mean(axis=0) # Mean activation per category (K)
    
    metadata = {}
    for i, act in enumerate(node_activations):
        metadata[i] = {"activation": float(act)}
        
    out_dir = "output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    output_file = os.path.join(out_dir, "igbundle_topology_dynamic.html")
    generate_viz(checkpoint_path, output_file, lite_mode=True, node_metadata=metadata)
    
    # Force forward slashes for the URL to avoid Windows platform issues
    url_path = output_file.replace("\\", "/")
    return f'<iframe src="file/{url_path}" width="100%" height="800px" style="border:none;"></iframe>'

def load_topo_stats():
    """Load statistics from thesis_stats.json for display."""
    stats_path = "thesis_stats.json"
    if not os.path.exists(stats_path):
        return [["Metric", "Value"], ["Status", "No stats found"]]
    try:
        import json
        with open(stats_path, 'r') as f:
            data = json.load(f)
        rows = [["Metric", "Value"]]
        for k, v in data.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    rows.append([f"{k}.{sk}", str(sv)])
            else:
                rows.append([k, str(v)])
        return rows
    except:
        return [["Error", "Could not parse metrics"]]

def launch_app(config_path, checkpoint_path):
    global MODEL, TOKENIZER
    MODEL, TOKENIZER = load_model(config_path, checkpoint_path)
    
    # Convert to standard Model Structure?
    # We need to attach hooks now that we have the model instance.
    hooks = HookManager(MODEL)
    hooks.attach()
    
    with gr.Blocks(title="ManifoldGL Explorer") as demo:
        gr.Markdown(
            """
            # 🌌 ManifoldGL: Geometric Bundle LLM Explorer
            **Model**: Qwen2.5-7B + IGBundle (Riemannian) | **Checkpoint**: Step 50
            
            Explore how the model adapts its semantic geometry in real-time.
            """
        )
        
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500, label="Hyperbolic Dialogue")
                    msg = gr.Textbox(placeholder="Ask me about the nature of reality...", label="User Input")
                    clear = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Geometric Telemetry")
                    curv_plot = gr.Image(label="Curvature Distribution $\sigma$")
                    aff_plot = gr.Image(label="Fiber Activation Map")
            
            def user(user_message, history):
                return "", history + [{"role": "user", "content": user_message}]

            def bot(history):
                user_message = history[-1]["content"]
                bot_message, p1, p2 = generate_with_viz(user_message, history[:-1])
                history.append({"role": "assistant", "content": bot_message})
                return history, p1, p2

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot], [chatbot, curv_plot, aff_plot]
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            
        with gr.Tab("System Architecture"):
            gr.Markdown(
                """
                ### IGBundle Topological View
                This view shows the 3D manifold geometry of the model. 
                Nodes represent semantic fibers, and their size/color reflects **real-time activations**.
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    refresh_btn = gr.Button("♻️ Refresh Topological State", variant="primary")
                    # Initial plot generation to avoid 404
                    initial_html = refresh_topology_view(checkpoint_path)
                    topo_display = gr.HTML(value=initial_html)
                with gr.Column(scale=1):
                    gr.Markdown("### Topological Statistics")
                    stats_table = gr.DataFrame(value=load_topo_stats(), interactive=False)
                    refresh_stats_btn = gr.Button("Refresh Stats")
                    
            refresh_btn.click(
                fn=lambda: refresh_topology_view(checkpoint_path),
                outputs=[topo_display]
            )
            refresh_stats_btn.click(
                fn=load_topo_stats,
                outputs=[stats_table]
            )

    # Broadly allow the current directory and its outputs
    app_root = os.getcwd()
    allowed = [app_root, os.path.join(app_root, "output")]
    
    demo.launch(
        share=False, 
        theme=gr.themes.Soft(), 
        allowed_paths=allowed
    )

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(base_dir, "configs", "qwen25_7b_igbundle_lora.yaml")
    
    # Try to find the latest checkpoint automatically or default to valid path
    default_checkpoint_dir = os.path.join(base_dir, "output", "igbundle_qwen7b_riemannian")
    # If checkpoint-100 exists, use it (Gold Master)
    if os.path.exists(os.path.join(default_checkpoint_dir, "checkpoint-100")):
        default_checkpoint = os.path.join(default_checkpoint_dir, "checkpoint-100")
    elif os.path.exists(os.path.join(default_checkpoint_dir, "checkpoint-50")):
        default_checkpoint = os.path.join(default_checkpoint_dir, "checkpoint-50")
    else:
        default_checkpoint = default_checkpoint_dir
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint) 
    args = parser.parse_args()
    
    launch_app(args.config, args.checkpoint)
