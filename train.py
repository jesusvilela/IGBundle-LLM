# === CRITICAL: Apply compatibility fixes BEFORE any other imports ===
import sys
import os as _os
_src_path = _os.path.join(_os.path.dirname(__file__), "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Disable TorchAO to prevent compatibility issues
_os.environ['DISABLE_TORCHAO'] = '1'
sys.modules['torchao'] = None # Force ImportError
# Memory fragmentation fix
_os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Apply Triton fixes for Windows
from igbundle.utils import triton_fix

# === Now safe to import everything else ===
import os
import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
# Check early for Windows to avoid Unsloth/Triton compatibility issues
if os.name == 'nt':
    print("Windows detected: Disabling Unsloth to avoid Triton errors.")
    HAS_UNSLOTH = False
else:
    try:
        from unsloth import FastLanguageModel
        HAS_UNSLOTH = True
    except ImportError:
        HAS_UNSLOTH = False

from igbundle.integrations.hf_patch import wrap_hf_candidate, StateCollector
from igbundle.modules.losses import SheafLoss

# Simple Config Object
class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

class SlowStepCallback(TrainerCallback):
    """Callback to prevent PSU/Thermal trips by pausing briefly between steps."""
    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(5.0) # Pause for 5 seconds to let hardware rest

class SaveAdapterCallback(TrainerCallback):
    """Callback to save IGBundle adapter weights at every save step."""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder, exist_ok=True)
            
        model = kwargs.get("model")
        # Unwrap if needed (Accelerated/Distributed)
        while hasattr(model, "module") or hasattr(model, "base_model") and not isinstance(model, torch.nn.Module): 
             # Careful unwrapping. PEFT wraps base_model.
             # Actually, checking keys on the top level 'model' state_dict should work if it recursively calls children.
             # But if it's wrapped in DDP, we need .module.
             if hasattr(model, "module"):
                 model = model.module
             else:
                 break
                 
        if model:
             # DEBUG: Print all keys first time to see structure
             # if state.global_step <= 10:
             #    print(f"[DEBUG KEYS] {list(model.state_dict().keys())[:10]}")
             
            adapter_sd = {k: v for k, v in model.state_dict().items() if "adapter" in k or "igbundle" in k}
            print(f"\n[SaveAdapterCallback] Found {len(adapter_sd)} adapter/igbundle keys.")
            
            if adapter_sd:
                save_path = os.path.join(checkpoint_folder, "adapter_weights.pt")
                torch.save(adapter_sd, save_path)
                print(f"[SaveAdapterCallback] Saved IGBundle weights to {save_path}")
            else:
                 # Fallback: Try to iterate modules manually if state_dict missing them
                 print("[SaveAdapterCallback] WARNING: No adapter keys in state_dict. Scanning named_parameters...")
                 # Manually build state dict
                 man_sd = {}
                 for n, p in model.named_parameters():
                     if "adapter" in n or "igbundle" in n:
                         man_sd[n] = p
                 if man_sd:
                     save_path = os.path.join(checkpoint_folder, "adapter_weights.pt")
                     torch.save(man_sd, save_path)
                     print(f"[SaveAdapterCallback] Saved IGBundle weights via named_parameters to {save_path}")
                 else:
                     print("[SaveAdapterCallback] ERROR: Could not find adapter weights even via named_parameters!")

class IGBundleTrainer(Trainer):
    def __init__(self, *args, state_collector=None, sheaf_loss_fn=None, lambda_glue=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_collector = state_collector
        self.sheaf_loss_fn = sheaf_loss_fn
        self.lambda_glue = lambda_glue
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Handle num_items_in_batch if present (ignored for now)
        # 1. Clear previous states
        if self.state_collector:
            self.state_collector.clear()
            
        # 2. Forward pass (LM loss calculated by model if labels provided)
        # We assume labels are in inputs
        outputs = model(**inputs)
        
        # Save past state if we need it? No.
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
        
        # 3. Add Aux losses
        aux_loss = 0.0
        if self.state_collector and self.lambda_glue > 0:
            states = self.state_collector.states
            # states is a list of MixtureStates from each layer
            # We can sum sheaf loss over layers or average
            layer_losses = []
            for s in states:
                l = self.sheaf_loss_fn(s)
                layer_losses.append(l)
            
            if layer_losses:
                aux_loss = torch.stack(layer_losses).mean() * self.lambda_glue
                
        # DEBUG: Print losses
        # print(f"LM Loss: {loss}, Aux Loss: {aux_loss}")
        # if torch.isnan(loss) or loss == 0.0:
        #    print(f"[DEBUG] Loss abnormal: {loss}. Aux: {aux_loss}")
                
        total_loss = loss + aux_loss
        
        # Thermal Management: Sleep longer to prevent reboots
        time.sleep(10.0)
        
        # Log auxiliary loss
        if self.state.global_step % self.args.logging_steps == 0:
            # This is a bit hacky to log inside compute_loss, but effective
            pass 

        return (total_loss, outputs) if return_outputs else total_loss

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen25_7b_igbundle_lora.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Config(cfg_dict)
    
    # Check CUDA early
    if not torch.cuda.is_available():
        print("CUDA missing. Switching base model to Qwen/Qwen2.5-0.5B for CPU compatibility.")
        cfg.base_model_id = "Qwen/Qwen2.5-0.5B"
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Model
    use_4bit = cfg.training.bf16 and torch.cuda.is_available() # Only use 4bit if cuda available
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Falling back to CPU and disabling 4-bit quantization.")
        # Override config for CPU
        cfg.base_model_id = "Qwen/Qwen2.5-0.5B" # Force smaller model
        cfg.training.bf16 = False
        use_4bit = False
        
    print(f"Loading {cfg.base_model_id} (Unsloth: {HAS_UNSLOTH})...")
    
    if HAS_UNSLOTH and torch.cuda.is_available():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = cfg.base_model_id,
            max_seq_length = 512,
            load_in_4bit = True,
            trust_remote_code = True,
        )
        # Apply LoRA via Unsloth
        model = FastLanguageModel.get_peft_model(
            model,
            r = cfg.lora.r,
            target_modules = cfg.lora.target_modules,
            lora_alpha = cfg.lora.lora_alpha,
            lora_dropout = cfg.lora.lora_dropout,
            bias = "none",
            use_gradient_checkpointing = "unsloth", # Optimized GC
            random_state = 3407,
        )
    else:
        # Fallback to standard loading
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Apply standard LoRA
        print("Applying standard LoRA...")
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["adapter"] # Ensure IGBundle weights are saved
        )
        model = get_peft_model(model, lora_config)

    # Disable use_cache for compatibility
    model.config.use_cache = False
    
    # 3. Wrap with IGBundle (Do this AFTER LoRA/Unsloth loading)
    print("Injecting IGBundle adapters...")
    # Overwrite hidden_size from model config to ensure matching shapes
    actual_model = model.model if hasattr(model, "model") else model
    if hasattr(actual_model.config, "hidden_size"):
        print(f"Detected model hidden_size: {actual_model.config.hidden_size}")
        cfg.ig_adapter.hidden_size = actual_model.config.hidden_size
        
    model = wrap_hf_candidate(model, cfg.ig_adapter)
    
    # Ensure IGBundle parameters are trainable (PEFT might have frozen them)
    print("Checking parameter freezing status...")
    count_trainable = 0
    for name, param in model.named_parameters():
        # Debug: Print first few params
        if count_trainable < 5:
             print(f"[DEBUG] Param: {name}, requires_grad: {param.requires_grad}")
             
        if "adapter" in name or "igbundle" in name or "lora" in name:
            param.requires_grad = True
            count_trainable += 1
            
    print(f"Forced unfreezing on {count_trainable} parameters match 'adapter'/'igbundle'/'lora'.")
            
    model.print_trainable_parameters()
    
    # 5. Setup Hooks & Loss
    collector = StateCollector()
    collector.attach(model)
    
    sheaf_loss = SheafLoss(cfg.loss.num_patches, cfg.ig_adapter.latent_dim, cfg.loss.tau)
    # Move loss to GPU
    if torch.cuda.is_available():
        sheaf_loss = sheaf_loss.to(torch.device("cuda"))
        
    model.sheaf_loss_module = sheaf_loss
    
    # 6. Data (Dataset loading)
    from transformers import DataCollatorForLanguageModeling
    from datasets import load_dataset
    
    if args.smoke_test:
        print("Smoke test: Using dummy data.")
        data = load_dataset("text", data_files={"train": ["dummy.txt"]}, split="train") # Hack or just generate memory dataset
        # Actually simplest memory dataset:
        from datasets import Dataset
        data = Dataset.from_dict({"text": ["Hello world " * 10] * 10})
    else:
        # Load Alpaca dataset
        print("Loading yahma/alpaca-cleaned...")
        data = load_dataset("yahma/alpaca-cleaned", split="train")
        
    # Format Alpaca into prompt
    def format_alpaca(example):
        if example.get("input", "") != "":
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n"
        
        # Add output for training
        text = prompt + example['output'] + tokenizer.eos_token
        return {"text": text}

    if "instruction" in data.column_names:
        print("Formatting Alpaca dataset...")
        data = data.map(format_alpaca)

    print("Tokenizing...")
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=512, # Qwen supports more but 512 is safe for 8GB VRAM
            padding=False, # DataCollator will pad
        )
        return outputs

    tokenized_datasets = data.map(
        tokenize, batched=True, remove_columns=data.column_names
    )
    
    # 7. Training Args
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=float(cfg.training.learning_rate),
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=getattr(cfg.training, 'max_steps', -1),
        logging_steps=cfg.training.logging_steps,
        bf16=cfg.training.bf16,
        optim=cfg.training.optim,
        max_grad_norm=getattr(cfg.training, 'max_grad_norm', 0.3), # Stability: Gradient clipping
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        label_names=["labels"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    
    trainer = IGBundleTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        state_collector=collector,
        sheaf_loss_fn=sheaf_loss,
        lambda_glue=float(cfg.loss.lambda_glue),
        callbacks=[SaveAdapterCallback(), SlowStepCallback()] # Re-enabled for stability
    )
    
    print("Starting training...")
    
    # Check for checkpoints
    last_checkpoint = None
    if os.path.isdir(cfg.training.output_dir):
        checkpoints = [d for d in os.listdir(cfg.training.output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = True
            print(f"Resuming from existing checkpoints in {cfg.training.output_dir}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print("Training complete.")
    # Save adapter + LoRA? PEFT saves LoRA.
    # We usually need to modify save logic to save IGBundle weights too since they are not standard PEFT.
    # But sticking to standard Trainer save might miss them if they are not in `peft_model`.
    # They are part of base `model` but not the `peft` wrapper's scope unless we are careful.
    # Actually `get_peft_model` wraps the base model.
    # Params with requires_grad=True should be saved if we use standard save? 
    # PEFT sets `disable_adapters=True` sometimes.
    
    # Manual save of state dict for safety
    if cfg.training.output_dir:
        torch.save(model.state_dict(), os.path.join(cfg.training.output_dir, "full_state_dict.pt"))

if __name__ == "__main__":
    train()
