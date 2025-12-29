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
from igbundle.integrations.hf_patch import wrap_hf_candidate

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

def launch_app(config_path, checkpoint_path):
    global MODEL, TOKENIZER
    MODEL, TOKENIZER = load_model(config_path, checkpoint_path)
    
    chat_interface = gr.ChatInterface(
        fn=generate_response,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Ask me anything...", container=False, scale=7),
        title="ManifoldGL: Geometric Bundle LLM",
        description="""
        **ManifoldGL**: LLM operating in layers of concave spaces using Information-Geometric Bundle Adapter.
        
        (c) Jesús Vilela Jato, all rights reserved.
        """,
        examples=["What is the nature of consciousness?", "Explain quantum entanglement.", "Write a python function to merge sort."],
        cache_examples=False,
    )
    
    chat_interface.launch(share=True)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(base_dir, "configs", "qwen25_7b_igbundle_lora.yaml")
    
    # Try to find the latest checkpoint automatically or default to valid path
    default_checkpoint_dir = os.path.join(base_dir, "output", "igbundle_qwen7b")
    # If checkpoint-50 exists, use it
    if os.path.exists(os.path.join(default_checkpoint_dir, "checkpoint-50")):
        default_checkpoint = os.path.join(default_checkpoint_dir, "checkpoint-50")
    else:
        default_checkpoint = default_checkpoint_dir
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint) 
    args = parser.parse_args()
    
    launch_app(args.config, args.checkpoint)
