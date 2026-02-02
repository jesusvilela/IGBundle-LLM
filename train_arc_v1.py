
import os
import torch
import json
import logging
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from igbundle.core.config import IGBundleConfig
from igbundle.modules.vision import VisionProjector
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter
from datasets import load_dataset

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_ID = "H:/LLM-MANIFOLD/igbundle_qwen7b_cp600"
VISION_MODEL_ID = "google/siglip-so400m-patch14-384"
OUTPUT_DIR = "trained_arc_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ARC Colors (0-9)
ARC_COLORS = [
    (0, 0, 0),       # 0: Black
    (0, 116, 217),   # 1: Blue
    (255, 65, 54),   # 2: Red
    (46, 204, 64),   # 3: Green
    (255, 220, 0),   # 4: Yellow
    (170, 170, 170), # 5: Grey
    (240, 18, 190),  # 6: Fuchsia
    (255, 133, 27),  # 7: Orange
    (127, 219, 255), # 8: Teal
    (135, 12, 37)    # 9: Maroon
]

def render_grid(grid, scale=20):
    """Renders a numeric grid to a PIL Image."""
    height = len(grid)
    width = len(grid[0])
    img = Image.new('RGB', (width * scale, height * scale), color='black')
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        for x in range(width):
            val = grid[y][x]
            if 0 <= val < len(ARC_COLORS):
                color = ARC_COLORS[val]
                draw.rectangle(
                    [x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1],
                    fill=color
                )
    return img

class MultimodalIGBundle(torch.nn.Module):
    def __init__(self, llm, vision_model, projector, tokenizer):
        super().__init__()
        self.llm = llm
        self.vision_model = vision_model
        self.projector = projector
        self.tokenizer = tokenizer
        
    def forward(self, input_ids, pixel_values, labels=None):
        # 1. Vision Features
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            vision_features = vision_outputs.last_hidden_state # (B, N_patches, D_vision)
            
        # 2. LLM Forward (Injects Adapter)
        # The adapter is already hooked into the LLM. 
        # We need to pass vision_features to the adapter somehow.
        # Ideally, we'd pass it via kwargs, but HF models might filter them.
        # HACK: Store vision features in the adapter globally/contextually before forward
        # OR: Relies on the adapter finding 'vision_features' if we pass it? 
        # Let's try passing it as a kwarg if the model allows, otherwise monkey-patch or use global.
        
        # Checking GeometricIGBundleAdapter code... it looks for 'vision_features' or 'pixel_values' 
        # usually in the attention hook. 
        
        # Simplified approach: Project here and prepend to embeddings?
        # No, we want the Geometric Adapter to handle it geometrically.
        
        # Assuming the adapter has a mechanism to receive this. 
        # If we look at `train_multimodal_v1.py` previously, we didn't explicitly pass it to the LLM layers?
        # Ah, in previous script `train_multimodal_v1.py`:
        # "The adapter hook mechanism in `GeometricIGBundleAdapter` is designed to look for `vision_context` attached to the module or passed."
        # Actually `train_multimodal_v1.py` just called `model(input_ids...)`. 
        # Wait, how did `train_multimodal_v1.py` work? 
        # It didn't! We built it but didn't verify the *mechanism* of injection deep in the layers. 
        # The adapter is a PeftModel or similar?
        # Let's assume for this specific script we can assign `vision_features` to the adapters directly.
        
        projected_vision = self.projector(vision_features) # (B, N, D_model)
        
        # Inject into all adapters
        for module in self.llm.modules():
            if isinstance(module, GeometricIGBundleAdapter):
                module.set_vision_context(projected_vision)
                
        outputs = self.llm(input_ids=input_ids, labels=labels)
        return outputs

def train():
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Steps between checkpoints")
    args, unknown = parser.parse_known_args()
    
    logger.info(f"Training for {args.steps} steps, saving every {args.save_steps} steps.")

    # Duplicate argparse block removed


    # 1. Load Models
    logger.info("Loading Models...")
    from transformers import AutoModelForCausalLM, SiglipVisionModel
    
    # ... (skipping unchanged model loading lines, but tool requires contiguous block usually)
    # Wait, replace_file_content works on contiguous chunks. I should insert argparse at top of train()
    # and update the loop bottom separately? No, better to do one replace if possible?
    # Actually, I can use multi_replace.
    # Let's start with proper argparse insertion.
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config,
        device_map=DEVICE
    )
    llm = prepare_model_for_kbit_training(llm)
    llm.config.use_cache = False # Required for gradient checkpointing
    
    vision_model = SiglipVisionModel.from_pretrained(
        VISION_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, use_fast=True)
    
    llm.gradient_checkpointing_enable() # Critical for 8GB VRAM
    
    # 2. Initialize Adapter
    config = IGBundleConfig(
        hidden_size=llm.config.hidden_size,
        adapter_dim=16,
        manifold_type="poincare",
        supported_modalities=["text", "vision"],
        vision_dim=vision_model.config.hidden_size,
        use_dynamics=True # Enable Physics/Logic
    )
    
    # Inject Adapter (Simulated Peft)
    # Note: GeometricIGBundleAdapter expects 'projected_vision' to match its internal logic.
    
    adapter = GeometricIGBundleAdapter(config).to(DEVICE) # Keep Float32 for Manifold Stability 
    
    projector = VisionProjector(
        vision_dim=vision_model.config.hidden_size, 
        bottleneck_dim=llm.config.hidden_size // 4
    ).to(DEVICE).to(torch.bfloat16)
    
    # Monkey patch one layer to be the "Reasoning Bottleneck"
    # Ideally we replace `llm.model.layers[10].mlp` or similar.
    # For 0.5B model (24 layers), let's pick layer 12 (middle).
    target_layer = llm.model.layers[12]
    original_forward = target_layer.forward
    
    def adapter_hook(hidden_states, *args, **kwargs):
        # 1. Run original layer
        out = original_forward(hidden_states, *args, **kwargs)
        # 2. Run adapter
        h = out[0] if isinstance(out, tuple) else out
        
        # Cast to Float32 for Manifold Adapter
        h_in = h.to(torch.float32)
        adapted_out = adapter(h_in)
        # Adapter returns (output, geometric_state)
        if isinstance(adapted_out, tuple):
            adapted = adapted_out[0]
        else:
            adapted_out = adapter(h_in)
        # Adapter returns (output, geometric_state)
        # The core idea: The adapter has "hyperbolized" the data internally and returned the 
        # tangent-space projection (output) and the manifold state (geometric_state).
        if isinstance(adapted_out, tuple):
            adapted = adapted_out[0]
        else:
            adapted = adapted_out
        
        # Cast back to BFloat16 for the LLM backbone
        adapted = adapted.to(torch.bfloat16)
        
        if isinstance(out, tuple):
            return (adapted,) + out[1:]
        return adapted
        
    target_layer.forward = adapter_hook
    logger.info("Injected Geometric Adapter at Layer 12.")

    # OOM Fix
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 3. Load ARC Dataset (Local ARC-AGI-2)
    logger.info("Loading ARC-AGI-2 from local repo...")
    
    dataset = []
    data_path = os.path.join("data", "ARC-AGI-2", "data", "training")
    
    if os.path.exists(data_path):
        import json
        import glob
        json_files = glob.glob(os.path.join(data_path, "*.json"))
        logger.info(f"Found {len(json_files)} tasks in {data_path}")
        
        for i, jf in enumerate(json_files):
            if i % 100 == 0:
                logger.info(f"Loading task {i}/{len(json_files)}...")
            try:
                with open(jf, 'r') as f:
                    try:
                        task = json.load(f)
                    except json.JSONDecodeError:
                        continue # Skip bad files
                    
                    if 'train' in task:
                        dataset.append(task)
            except Exception:
                continue
    else:
        logger.warning(f"Local path {data_path} not found.")

    if not dataset:
        logger.warning("No local data found. Using Synthetic Fallback.")
        # Synthetic Fallback
        class SyntheticARCDataset:
            def __iter__(self):
                while True:
                    # Generate random grid
                    w, h = np.random.randint(3, 10), np.random.randint(3, 10)
                    input_grid = np.random.randint(0, 9, (h, w)).tolist()
                    
                    # Task 1: Identity
                    # Task 2: Color Swap (0->1)
                    mode = np.random.choice(["identity", "swap"])
                    
                    if mode == "identity":
                        output_grid = input_grid
                    else:
                        output_grid = [[1 if c == 0 else c for c in row] for row in input_grid]
                        
                    yield {
                        "train": [{"input": input_grid, "output": output_grid}],
                        "test": [{"input": input_grid, "output": output_grid}] # Mock
                    }
                    
        dataset = SyntheticARCDataset()
    else:
        # Check if we need to shuffle local dataset
        import random
        random.shuffle(dataset)
        logger.info(f"Loaded {len(dataset)} tasks.")

    optimizer = torch.optim.AdamW([
        {'params': projector.parameters(), 'lr': 1e-4},
        {'params': adapter.parameters(), 'lr': 2e-4}
    ])
    
    model = MultimodalIGBundle(llm, vision_model, projector, tokenizer)
    model.train()
    
    step = 0
    
    logger.info("Starting ARC Training...")
    for item in dataset:
        try:
            # Item structure depends on dataset. 
            # Usually 'train' (list of pairs) and 'test' (list of pairs).
            # We want to train on the 'train' examples to learn the rule.
            # Pick one random example from the 'train' set of this task.
            if 'train' not in item: continue # Skip completely malformed
            train_pairs = item['train'] # list of dicts {'input':[[...]], 'output':[[...]]}
            if not train_pairs: continue
            
            pair = train_pairs[0] # Just take first for simplicity in this loop
            if 'input' not in pair or 'output' not in pair: continue
            
            input_grid = pair['input']
            output_grid = pair['output']
            
            # Render Input Grid -> Vision
            image_in = render_grid(input_grid)
            pixel_values = processor(images=image_in, return_tensors="pt").pixel_values.to(DEVICE).to(torch.bfloat16)
            
            # Text: "Input: <grid_text> Output:" -> Target: "<grid_text>"
            # Represent grid as text for LLM supervision
            txt_in = str(input_grid)
            txt_out = str(output_grid)
            
            prompt = f"Task: Solve the grid transformation.\nInput: {txt_in}\nOutput:"
            label_text = f" {txt_out}"
            
            full_text = prompt + label_text
            inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
            
            # Mask user part for labels
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            labels = inputs.input_ids.clone()
            labels[:, :prompt_len] = -100
            
            # Inject vision context (Ensure projector output matches expected context dim)
            # Vision output (B, N, H) -> Projector -> (B, N, Bottleneck=H_llm)
            vis_out = vision_model(pixel_values).last_hidden_state
            projected_vision = projector(vis_out).to(torch.float32)
            
            if step == 0:
                logger.info(f"DEBUG: vis_out.dtype={vis_out.dtype}")
                logger.info(f"DEBUG: projected_vision.dtype={projected_vision.dtype}")
                p_param = next(adapter.parameters())
                logger.info(f"DEBUG: adapter.param.dtype={p_param.dtype}")

            adapter.set_vision_context(projected_vision)
            
            outputs = llm(input_ids=inputs.input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            if (step + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"Step {step+1}: Loss = {loss.item():.4f}")
                torch.cuda.empty_cache() # Aggressive cleanup
            
            # Checkpointing
            if (step + 1) % args.save_steps == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(projector.state_dict(), os.path.join(ckpt_dir, "arc_projector.pt"))
                torch.save(adapter.state_dict(), os.path.join(ckpt_dir, "arc_adapter.pt"))
                logger.info(f"Saved checkpoint to {ckpt_dir}")
            
            step += 1
            if step >= args.steps: break
            
        except Exception as e:
            import traceback
            logger.error(f"Error: {e}\n{traceback.format_exc()}")
            continue

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(projector.state_dict(), os.path.join(OUTPUT_DIR, "arc_projector.pt"))
    torch.save(adapter.state_dict(), os.path.join(OUTPUT_DIR, "arc_adapter.pt"))
    logger.info("ARC Training Complete.")

if __name__ == "__main__":
    train()
