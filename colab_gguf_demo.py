# ==============================================================================
# 🌌 ManifoldGL: Colab EPIC Demo (v8 - Neural Glass Edition)
# ==============================================================================

import subprocess
import sys
import os

# --- Configuration ---
HF_REPO_ID = "jesusvilela/igbundle-qwen2.5-7b-riemannian"
MODEL_FILENAME = "igbundle_qwen7b_riemannian.gguf" 

print("🚀 Starting EPIC Environment Setup...")

def run_cmd(cmd_list):
    print(f"👉 Running: {' '.join(cmd_list)}")
    subprocess.check_call(cmd_list)

def install(package, args=None):
    cmd = [sys.executable, "-m", "pip", "install", package]
    if args: cmd.extend(args)
    run_cmd(cmd)

# --- 1. Hardware Detection ---
HAS_GPU = False
GPU_NAME = "CPU"
try:
    GPU_NAME = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding="utf-8").strip()
    print(f"✅ NVIDIA GPU Detected: {GPU_NAME}")
    HAS_GPU = True
except:
    print("⚠️ No NVIDIA GPU detected. Switching to CPU Fallback.")
    HAS_GPU = False

# --- 2. Install Build Tools ---
print("📦 Installing Core Dependencies...")
try:
    import cmake
except ImportError:
    install("cmake")
    install("scikit-build-core")
    install("ninja")

try:
    import gradio as gr
    from huggingface_hub import hf_hub_download
    import matplotlib
    import numpy
except ImportError:
    install("gradio")
    install("huggingface_hub")
    install("matplotlib")
    install("numpy")

# --- 3. Install Engine ---
print("⚙️ Checking Inference Engine...")
try:
    from llama_cpp import Llama
    print("✅ Engine already active.")
except ImportError:
    if HAS_GPU:
        print(f"🚀 Compiling Optimized Engine for {GPU_NAME}...")
        os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on -DGGML_FLASH_ATTN=on"
        install("llama-cpp-python", ["--no-cache-dir", "--force-reinstall", "--upgrade", "--verbose"])
    else:
        print("🐌 Compiling Safe-Mode Engine (CPU)...")
        if "CMAKE_ARGS" in os.environ: del os.environ["CMAKE_ARGS"]
        install("llama-cpp-python", ["--no-cache-dir", "--force-reinstall", "--upgrade"])
    from llama_cpp import Llama

# --- 4. Custom CSS (Neural Glass) ---
custom_css = """
body { background: #0f0c29; background: linear-gradient(to right, #24243e, #302b63, #0f0c29); color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
.gradio-container { background: rgba(0,0,0,0) !important; max-width: 1200px !important; }
h1, h2, h3 {  background: -webkit-linear-gradient(#00c6ff, #0072ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.tensor-plot { border: 1px solid rgba(0, 198, 255, 0.3); border-radius: 10px; background: rgba(0,0,0,0.4); box-shadow: 0 0 15px rgba(0, 198, 255, 0.1); }
button.primary { background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%) !important; border: none !important; box-shadow: 0 0 10px #0072ff !important; transition: all 0.3s ease; }
"""

# --- 5. Load Model & UI Logic ---
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download
import gradio as gr

print(f"📥 Downloading Model: {MODEL_FILENAME}...")
try:
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    n_gpu_layers = -1 if HAS_GPU else 0
    print(f"🧠 Loading Neural Network (Offload: {n_gpu_layers})...")
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=4096, verbose=False)
except Exception as e:
    print(f"\n❌ FATAL LOAD ERROR: {e}")
    sys.exit(1)

def generate_telemetry(text_len):
    """Simulates geometric telemetry."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='none')
    
    steps = np.arange(0, 100)
    curvature = -1.0 + (0.5 * np.sin(steps * 0.1)) + np.random.normal(0, 0.05, 100)
    
    ax1.plot(steps, curvature, color='#00c6ff', linewidth=2)
    ax1.set_title("Ricci Curvature κ", color='white')
    ax1.set_ylim(-1.5, -0.5)
    ax1.set_facecolor('none')
    ax1.tick_params(colors='white')
    ax1.grid(color='#333', linestyle='--')
    for spine in ax1.spines.values(): spine.set_edgecolor('#444')

    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.5 + 0.1 * np.cos(5*theta + text_len/10)
    ax2.plot(r * np.cos(theta), r * np.sin(theta), color='#bd34fe', linewidth=2)
    ax2.fill(r * np.cos(theta), r * np.sin(theta), color='#bd34fe', alpha=0.3)
    ax2.set_title("Fiber Holonomy", color='white')
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_aspect('equal')
    ax2.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return PILImage.open(buf)

def chat_stream(message, history):
    full_prompt = ""
    for msg in history:
        role = msg['role']
        content = msg['content']
        if role == "user": full_prompt += f"### Instruction:\n{content}\n\n"
        elif role == "assistant": full_prompt += f"### Response:\n{content}\n\n"
    full_prompt += f"### Instruction:\n{message}\n\n### Response:\n"
    
    response_text = ""
    stream = llm(full_prompt, max_tokens=512, stop=["### Instruction:", "### End"], echo=False, stream=True)
    
    for output in stream:
        token = output['choices'][0]['text']
        response_text += token
        yield response_text, None 
        
    viz = generate_telemetry(len(response_text))
    yield response_text, viz

# --- UI Construction ---
with gr.Blocks(title="ManifoldGL EPIC", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 ManifoldGL: Neural Glass (GGUF)")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, type="messages", label="Manifold Interface")
            msg = gr.Textbox(placeholder="Enter abstract reasoning task...", label="Input Protocol", container=False)
            with gr.Row():
                btn_run = gr.Button("Initialize Generation", variant="primary", elem_classes=["primary"])
                btn_clear = gr.Button("Flush Buffers")

        with gr.Column(scale=1):
            gr.Markdown("### 🧠 BrainTop Telemetry")
            telemetry_plot = gr.Image(label="Live Manifold State (Simulated)", elem_classes=["tensor-plot"])

    chat_history = gr.State([])

    def user(user_message, history):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history):
        user_msg = history[-1]['content']
        history_context = history[:-1]
        for text_chunk, image_update in chat_stream(user_msg, history_context):
            updated_history = history + [{"role": "assistant", "content": text_chunk}]
            yield updated_history, image_update

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot], [chatbot, telemetry_plot])
    btn_run.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot], [chatbot, telemetry_plot])
    btn_clear.click(lambda: [], None, chatbot, queue=False)

print("🚀 Launching Neural Glass Interface...")
demo.queue()
demo.launch(share=True, debug=True)
