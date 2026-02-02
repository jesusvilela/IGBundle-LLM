#!/usr/bin/env python3
"""
IGBundle Multimodal Training Script v1.0
Integrates SigLIP vision encoder + VisionProjector + GeometricAdapter.
"""

import os
import sys
import torch
import json
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel

# Setup path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from igbundle.modules.vision import VisionProjector
from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter
from igbundle.integrations.hf_patch import wrap_hf_candidate
from igbundle.core.config import IGBundleConfig
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalIGBundle(torch.nn.Module):
    """Wrapper that handles Vision -> Projection -> LLM Embedding Injection"""
    def __init__(self, llm, vision_model, projector, tokenizer):
        super().__init__()
        self.llm = llm
        self.vision_model = vision_model
        self.projector = projector
        self.tokenizer = tokenizer
        
    def forward(self, input_ids, pixel_values=None, labels=None):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # If we have visual input, inject it
        if pixel_values is not None:
            # 1. Encode Vision
            with torch.no_grad(): # Keep vision model frozen for now
                vision_out = self.vision_model(pixel_values=pixel_values)
                image_features = vision_out.last_hidden_state
                
            # 2. Project to Manifold
            # (Batch, Patches, VisionDim) -> (Batch, Patches, LLMDim)
            projected_features = self.projector(image_features)
            
            # 3. Naive Injection (Prepend to embeddings)
            # We assume the <image> token is at start. Real impl would mask carefully.
            # Here we just prepend visual features to text embeddings.
            inputs_embeds = torch.cat([projected_features, inputs_embeds], dim=1)
            
            # Adjust labels for padding (ignore visual tokens for loss)
            if labels is not None:
                ignore_tokens = torch.full((labels.shape[0], projected_features.shape[1]), -100, device=labels.device)
                labels = torch.cat([ignore_tokens, labels], dim=1)
                
        # 4. LLM Forward
        return self.llm(inputs_embeds=inputs_embeds, labels=labels)

def train():
    MODEL_ID = "Qwen/Qwen2.5-0.5B" # Fast iteration
    VISION_ID = "google/siglip-so400m-patch14-384"
    DATA_PATH = "data/physics_dynamics.jsonl"
    OUTPUT_DIR = "trained_physics_adapter"
    
    logger.info("Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
    
    vision_model = SiglipVisionModel.from_pretrained(VISION_ID, device_map="auto", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(VISION_ID)
    
    # 1. Initialize Projector (Vision Dim -> LLM Dim)
    vision_dim = vision_model.config.hidden_size # 1152 for SigLIP
    llm_dim = llm.config.hidden_size
    projector = VisionProjector(vision_dim, llm_dim).to(llm.device).to(torch.bfloat16)
    
    # 2. Apply Geometric Adapter to LLM
    from igbundle.core.config import IGBundleConfig
    geo_config = IGBundleConfig(hidden_size=llm_dim, latent_dim=16, supported_modalities=["text", "vision"], use_dynamics=True) 
    llm = wrap_hf_candidate(llm, geo_config, adapter_class=GeometricIGBundleAdapter)
    
    # Cast Adapter to Float32 for Geometric Precision (Fixes Dtype Mismatch)
    for name, module in llm.named_modules():
        if "adapter" in name.lower() or "geometric" in name.lower():
            module.to(torch.float32)
            
    # 3. Wrap Module
    model = MultimodalIGBundle(llm, vision_model, projector, tokenizer)
    
    # 4. Optimizer (Train Adapter + Projector)
    optimizer = torch.optim.AdamW([
        {'params': projector.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in llm.named_parameters() if "adapter" in n and p.requires_grad], 'lr': 2e-4}
    ])
    
    # 5. Load HF ScienceQA Dataset (Streaming)
    from datasets import load_dataset
    import io
    logger.info("Loading ScienceQA dataset (Streaming)...")
    dataset = load_dataset("derek-thomas/ScienceQA", split="train", streaming=True)
    
    # Filter for Physics/Natural Science with Images
    def is_physics_vision(example):
        return example['image'] is not None and example['topic'] in ['natural science', 'physics']
    
    filtered_dataset = dataset.filter(is_physics_vision).take(200) # Train on 200 relevant examples for this dev cycle
    
    model.train()
    accumulation_steps = 4 
    optimizer.zero_grad()
    
    step = 0
    total_loss = 0
    
    logger.info("Starting robust training loop...")
    for item in filtered_dataset:
        try:
            # Prepare Text
            question = item['question']
            choices = item['choices']
            answer_idx = item['answer']
            explanation = item['solution']
            # Format: User: <Q> + <Choices> \n Assistant: <Answer> + <Explanation>
            prompt = f"User: {question}\nOptions: {choices}\nAssistant: The answer is {choices[answer_idx]}. {explanation}"
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(llm.device)
            
            # Prepare Image
            image_data = item['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            else:
                image = image_data # Already PIL or handled otherwise
                
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(llm.device).to(torch.bfloat16)
            
            # Forward
            outputs = model(input_ids=inputs.input_ids, pixel_values=pixel_values, labels=inputs.input_ids)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"Step {step+1}: Loss = {total_loss / accumulation_steps:.4f}")
                total_loss = 0
                
            step += 1
            if step >= 100: # Cap at 100 steps for this dev cycle
                break
                
        except Exception as e:
            logger.warning(f"Skipping sample due to error: {e}")
            continue
            
    # Save adapter
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(projector.state_dict(), os.path.join(OUTPUT_DIR, "vision_projector.pt"))
    adapter_state = {n: p for n, p in llm.named_parameters() if "adapter" in n}
    torch.save(adapter_state, os.path.join(OUTPUT_DIR, "geometric_adapter.pt"))
    logger.info("Training Complete (ScienceQA Physics). Weights saved.")

if __name__ == "__main__":
    train()
