try:
    from igbundle.utils import triton_fix
except ImportError:
    # If package not installed in editable mode, try relative or local path
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from igbundle.utils import triton_fix

import os
import argparse
import yaml
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image as PILImage

# Global Viz State
VIZ_STATE = {
    "curvature": [],
    "affinity": []
}

class HookManager:
    """Manages forward hooks to capture intermediate geometric states."""
    def __init__(self, model):
        self.hooks = []
        self.model = model
        
    def _curvature_hook(self, module, input, output):
        # Expected output from ManifoldKernel: (projected, curvature, ...)
        # Or if it's the Adapter output...
        # Let's inspect the module structure. 
        # For now, we assume we hook into the 'igbundle_adapter' layers.
        # If output is a tuple and len > 1, 2nd element is usually curvature/aux.
        if isinstance(output, tuple) and len(output) > 1:
            # We treat the second element as curvature scalar field if shape matches
            curv = output[1]
            if isinstance(curv, torch.Tensor):
                VIZ_STATE["curvature"].append(curv.detach().cpu().numpy())

    def _affinity_hook(self, module, input, output):
        # Attention weights are tricky to hook in PEFT.
        # We might need to rely on the model returning attentions.
        pass

    def attach(self):
        # Find all IGBundle modules
        for name, module in self.model.named_modules():
            if "igbundle" in name.lower() and "kernel" in name.lower():
                h = module.register_forward_hook(self._curvature_hook)
                self.hooks.append(h)
                print(f"Hooked {name} for curvature.")

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
def plot_curvature():
    """Generate a plot of the curvature distribution."""
    if not VIZ_STATE["curvature"]:
        return None
    
    data = np.concatenate([c.flatten() for c in VIZ_STATE["curvature"]])
    
    plt.figure(figsize=(8, 4))
    sns.histplot(data, bins=30, kde=True, color="purple")
    plt.title("Curvature $\sigma(x)$ Distribution (Riemannian Dispersion)")
    plt.xlabel("Local Curvature Value")
    plt.ylabel("Frequency")
    plt.axvline(x=-1.0, color='r', linestyle='--', label="Hyperbolic Ideal (-1)")
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
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
            
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
    else:
        print("CUDA not available. Loading in float32 on CPU.")
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
    
    print(f"Loading LoRA/Adapter from {checkpoint_path}")
    print(f"Loading LoRA/Adapter from {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        try:
            # Robust Loading Strategy:
            # 1. Init PeftModel with config only (no weights yet)
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            model = PeftModel(model, peft_config)
            
            # 2. Load weights manually with strict=False
            # This allows us to load the standard LoRA weights (lora_A, lora_B) 
            # while ignoring potential mismatches or custom IGBundle parameters that PEFT might choke on.
            if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
                from safetensors.torch import load_file
                adapters_weights = load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
                msg = model.load_state_dict(adapters_weights, strict=False)
                print(f"LoRA weights loaded (strict=False). Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            elif os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
                adapters_weights = torch.load(os.path.join(checkpoint_path, "adapter_model.bin"), map_location="cpu")
                msg = model.load_state_dict(adapters_weights, strict=False)
                print(f"LoRA weights loaded from bin (strict=False). Missing: {len(msg.missing_keys)}")
            else:
                 print("No adapter_model file found.")
                 
        except Exception as e:
            print(f"Warning: Could not load LoRA via PeftModel manual load ({e}). Continuing with base model + initialized adapter.")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Running with initialized adapter (untrained).")
        
    # Attempt to load full state dict for adapter params if available
    full_sd_path = os.path.join(checkpoint_path, "full_state_dict.pt")
    adapter_w_path = os.path.join(checkpoint_path, "adapter_weights.pt")
    
    adapter_sd = None
    if os.path.exists(full_sd_path):
        print(f"Loading adapter weights from {full_sd_path}")
        sd = torch.load(full_sd_path, map_location="cpu")
        adapter_sd = {k: v for k, v in sd.items() if "igbundle" in k or "adapter" in k}
    elif os.path.exists(adapter_w_path):
        print(f"Loading adapter weights from {adapter_w_path}")
        adapter_sd = torch.load(adapter_w_path, map_location="cpu")
        
    if adapter_sd:
        model.load_state_dict(adapter_sd, strict=False)
        print("Adapter weights loaded.")
    else:
        print("No full_state_dict.pt or adapter_weights.pt found. Adapter might be untrained initialized!")
        
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
    
    return response, curv_plot

def launch_app(config_path, checkpoint_path):
    global MODEL, TOKENIZER
    MODEL, TOKENIZER = load_model(config_path, checkpoint_path)
    
    # Convert to standard Model Structure?
    # We need to attach hooks now that we have the model instance.
    hooks = HookManager(MODEL)
    hooks.attach()
    
    with gr.Blocks(title="ManifoldGL Explorer", theme=gr.themes.Soft()) as demo:
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
                    curv_plot = gr.Plot(label="Curvature Distribution $\sigma$")
                    # affinity_plot = gr.Plot(label="Fiber Affinity Matrix") # Placeholder
            
            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                user_message = history[-1][0]
                bot_message, plot = generate_with_viz(user_message, history[:-1])
                history[-1][1] = bot_message
                return history, plot

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [chatbot], [chatbot, curv_plot]
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            
        with gr.Tab("System Architecture"):
            gr.Markdown("### IGBundle Topological View")
            gr.HTML('<iframe src="file/output/igbundle_topology_riemannian.html" width="100%" height="600px"></iframe>')

    demo.launch(share=True)

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
