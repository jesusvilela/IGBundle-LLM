"""Download-free CPU optimizer check for the public Qwen2 adapter path."""

import json
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from igbundle.integrations.hf_patch import wrap_hf_candidate


def run(steps):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        attention_dropout=0., pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        lora_dropout=0., task_type="CAUSAL_LM",
    ))
    adapter_config = SimpleNamespace(
        hidden_size=16, bottleneck_dim=4, num_components=2, latent_dim=2,
        num_categories=2, alpha=1., beta=1., eta_f=.1, eta_b=.1, eta_w=.1,
        adapter_scale=.1, dropout=0.,
    )
    model = wrap_hf_candidate(model, adapter_config).train()
    parameters = dict(model.named_parameters())
    before = {name: value.detach().clone() for name, value in parameters.items()}
    optimizer = torch.optim.AdamW([p for p in parameters.values() if p.requires_grad], lr=.01)
    tokens = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 9, 2]])
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CPU smoke training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    changed = {name for name, value in parameters.items() if not torch.equal(value, before[name])}
    adapter_updates = sum(".adapter." in name for name in changed)
    lora_updates = sum("lora_" in name for name in changed)
    frozen_updates = sum(not parameters[name].requires_grad for name in changed)
    if not adapter_updates or not lora_updates or frozen_updates:
        raise RuntimeError("Expected both adapter groups to update while the base stays frozen")
    print(json.dumps({
        "device": "cpu", "steps": steps, "losses": losses,
        "adapter_tensors_updated": adapter_updates, "lora_tensors_updated": lora_updates,
        "frozen_tensors_updated": frozen_updates,
    }))
    print("Training complete.")
