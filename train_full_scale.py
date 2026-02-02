
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

from igbundle.modules.geometric_adapter import create_geometric_adapter
from igbundle.core.config import IGBundleConfig

# --- Environment Setup (Force H: Drive for Cache & Temp) ---
os.environ["HF_HOME"] = "H:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "H:/hf_cache/datasets"
os.environ["TORCH_HOME"] = "H:/hf_cache/torch"
os.environ["TMP"] = "H:/tmp"
os.environ["TEMP"] = "H:/tmp"
os.environ["TMPDIR"] = "H:/tmp"

# --- Configuration ---
MODEL_ID = "h:/LLM-MANIFOLD/igbundle_qwen7b_cp600" # Absolute path to base model
DATASET_PATH = "igbundle-llm/data/full_scale_train.jsonl"
OUTPUT_DIR = "igbundle_full_scale_reasoning"

# Training Hyps
MAX_SEQ_LENGTH = 1024 # Trade-off for VRAM
BATCH_SIZE = 1       # Per device (Reduced from 4 to fit VRAM)
GRAD_ACCUM = 32      # Effective BS = 32
LEARNING_RATE = 2e-5 # LLM LR
GEO_LR_SCALE = 1.0   # Geometry learns faster (1e-4) -> Reduced to 1.0 for Stability
NUM_EPOCHS = 1
SAVE_STEPS = 20      # Aggressive checkpointing (Every ~2 hours) to prevent data loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FullScaleTrainer:
    def __init__(self):
        # 1. Manifold Configuration (KAN Geometry)
        self.config = IGBundleConfig(
            hidden_size=3584, # Qwen 7B
            num_components=8, 
            latent_dim=64,    
            num_categories=16,
            use_dynamics=True,          # Phase 2: Hamiltonian Dynamics
            use_geodesic_attn=True,     # Phase 2: Geodesic Attention
            manifold_type="kan",        # Phase 3.5: Learnable KAN Manifold
            adapter_scale=0.1
        )
        self.accelerator = Accelerator()
        
    def setup_model(self):
        print(f"Loading Base Model: {MODEL_ID}")
        
        # 4-Bit Quantization Config (Critical for 8GB VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Match training precision
            bnb_4bit_use_double_quant=True,
        )
        
        # Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, # Load in BF16
            device_map="auto" # Now fits in GPU
        )
        self.model.config.use_cache = False # Required for gradient checkpointing/training
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize Geometric Adapter (KAN)
        print("Initializing KAN Adapter...")
        # Keep adapter in FP32 for numerical stability (Riemannian math explodes in FP16)
        # Inputs will be cast up to FP32 in the hook
        self.adapter = create_geometric_adapter(self.config).to(DEVICE)
        
        # Inject Adapter (Hook-based)
        # We hook into the mid-layer or last layer.
        # For Deep Reasoning, we want to intercept high-level thought process.
        # Let's hook into layer 20 (Deep enough for semantics, early enough for influence)
        target_layer = self.model.model.layers[20]
        
        def adapter_hook(module, args, output):
            # output is tuple (hidden_states, ...) or Tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output 
            # Ensure dtype compatibility
            target_dtype = self.adapter.input_proj.weight.dtype
            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(target_dtype)
                
            # Apply Geometric Transformation
            new_hidden, geo_state = self.adapter(hidden_states)
            
            # Store geo_state for loss computation (Thread-local or attribute hack)
            # Since we are in a forward pass, we can store it on the module temporarily
            self._current_geo_state = geo_state
            
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            else:
                return new_hidden
            
        self.hook_handle = target_layer.register_forward_hook(adapter_hook)
        print("Adapter Injected at Layer 20.")

    def prepare_dataset(self):
        print(f"Loading Dataset: {DATASET_PATH}")
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
        
        def process(examples):
            # Chat Format: System + User + Assistant
            texts = []
            for inp, out in zip(examples['input'], examples['output']):
                chat = [
                    {"role": "system", "content": "You are a deep reasoning engine capable of complex mathematical and logical thought."},
                    {"role": "user", "content": inp},
                    {"role": "assistant", "content": out}
                ]
                text = self.tokenizer.apply_chat_template(chat, tokenize=False)
                texts.append(text)
            return {"text": texts}
            
        self.dataset = dataset.map(process, batched=True, remove_columns=dataset.column_names, keep_in_memory=True)
        print(f"Dataset Prepared: {len(self.dataset)} samples.")

    def train(self):
        print("Starting Full Scale Training...")
        
        # Use SFTConfig for TRL 0.24.0+
        training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            fp16=False,
            bf16=True,   # Switch to BF16 for stability (Wide dynamic range prevents NaN)
            logging_steps=10,
            save_steps=SAVE_STEPS,
            save_total_limit=3, # Keep only last 3 checkpoints to save space
            optim="adamw_torch",
            report_to="tensorboard",
            remove_unused_columns=False,
            max_length=MAX_SEQ_LENGTH,   # Renamed from max_seq_length per TRL 0.24.0 error
            dataset_text_field="text",
            packing=False,
            max_grad_norm=0.3  # prevent exploding gradients
        )
        
        # Prepare for k-bit training (Gradient Checkpointing + Cast Output)
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA (QLoRA) to enable gradients on 4-bit model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Custom Trainer to add Geometric Loss
        class GeoSFTTrainer(SFTTrainer):
            # Remove custom __init__ to avoid signature conflicts with various trl versions
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                # Standard LLM Loss (CrossEntropy)
                
                # Actual Loss Calculation
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
                
                # Add Geometric Losses
                # Retrieve state from the hook interception stored in the MODEL (or Trainer if injected)
                
                # We expect the trainer to have 'geo_adapter' and 'parent_trainer' injected after init
                if hasattr(self, 'geo_adapter'):
                    if hasattr(self, 'parent_trainer') and hasattr(self.parent_trainer, '_current_geo_state'):
                        geo_state = self.parent_trainer._current_geo_state
                        
                        # Compute Geometric Losses
                        geo_losses = self.geo_adapter.compute_geometric_losses(geo_state)
                    
                        # Log them
                        if self.state.global_step % 10 == 0:
                            loss_log = {f"geo/{k}": v.item() for k,v in geo_losses.items()}
                            loss_log["llm_loss"] = loss.item()
                            print(f"Step {self.state.global_step}: LLM={loss.item():.4f} " + 
                                  f"GeoH={geo_losses.get('hamiltonian_energy', 0):.4f}")
                        
                        # Weighted Sum
                        # Geo Loss should be small regularization
                        total_geo_loss = sum(geo_losses.values())
                        loss += total_geo_loss
                    
                return (loss, outputs) if return_outputs else loss
        
        # We need to register adapter params. 
        # SFTTrainer takes 'model'. We can patch the model to include adapter params?
        # Or Just loop over them.
        # Better: Add adapter as a submodule of model?
        # model.geometric_adapter = self.adapter
        self.model.geometric_adapter = self.adapter # Register as module
        
        trainer = GeoSFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            processing_class=self.tokenizer, # Renamed from tokenizer for TRL 0.24.0
            args=training_args, # SFTConfig passed here
        )
        # Inject dependencies
        trainer.geo_adapter = self.adapter
        trainer.parent_trainer = self
        
        # Resume from checkpoint if it exists
        checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoint = True # TRL will auto-find the latest
                print(f"Resuming from latest checkpoint in {OUTPUT_DIR}")
        
        trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save Final
        trainer.save_model()
        torch.save(self.adapter.state_dict(), f"{OUTPUT_DIR}/geometric_adapter_final.pt")
        print(f"Training Complete. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    trainer = FullScaleTrainer()
    trainer.setup_model()
    trainer.prepare_dataset()
    trainer.train()
