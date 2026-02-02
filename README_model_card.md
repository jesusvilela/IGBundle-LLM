
---
license: apache-2.0
tags:
- geometry
- riemannian
- qwen2.5
- fine-tuned
- manifold-learning
base_model: unsloth/Qwen2.5-7B-Instruct
library_name: peft
---

# IGBundle-Qwen2.5-7B-Riemannian

This model is a fine-tuned version of **Qwen2.5-7B-Instruct** incorporating **Riemannian Geometric Regularization** (IGBundle framework).
It was trained to learn manifold structures within the latent space, preserving fiber bundle topology and minimizing curvature deviation.

## Model Details
- **Base Model**: `unsloth/Qwen2.5-7B-Instruct`
- **Method**: Riemannian Manifold Fine-Tuning (IGBundle) with LoRA + GeometricAdapter.
- **Precision**: BFloat16
- **Context Length**: 32k (trained/eval at 4k-8k)

## Training & Performance
- **Training Steps**: 700 (Resumed + 100 Geometric Steps)
- **Objective**: Causal LM Loss + Curvature Loss + Bundle Consistency Loss
- **Optimization**: RiemannianOptimizer (Natural Gradients)

## Evaluation (ARC-AGI)
Evaluated on the ARC-AGI Challenge validation set (subset).

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "jesusvilela/igbundle-qwen2.5-7b-riemannian"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Analyze the geometric structure of this problem..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```
