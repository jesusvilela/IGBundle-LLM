"""Offline CPU training and strict export of the public Qwen2 adapter pair."""

import json
from itertools import cycle, islice
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from igbundle.integrations.checkpoint import load_peft_checked, load_weights
from igbundle.integrations.hf_patch import wrap_hf_candidate
from igbundle.modules.adapter import IGBundleAdapter


def read_examples(path, tokenizer, max_length):
    examples = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON on line {number}") from error
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Expected nonempty text on line {number}")
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            if len(tokens) < 2:
                raise ValueError(f"Expected at least two tokens on line {number}")
            examples.append(torch.tensor([tokens], dtype=torch.long))
    if not examples:
        raise ValueError("Training JSONL contains no examples")
    return examples


def load_base(checkpoint):
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True, dtype=torch.float32,
    )
    if not isinstance(base, Qwen2ForCausalLM):
        raise ValueError("Local training requires a Qwen2 causal language model")
    return base


def fit_and_save(checkpoint, examples, steps, directory):
    base = load_base(checkpoint)
    layers = base.model.layers
    hidden = base.config.hidden_size
    adapter_config = SimpleNamespace(
        hidden_size=hidden, bottleneck_dim=min(32, hidden), num_components=2,
        num_categories=4, latent_dim=4, alpha=1., beta=1., eta_f=.1,
        eta_b=.1, eta_w=.1, adapter_scale=.1, dropout=0.,
    )
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        lora_dropout=0., task_type="CAUSAL_LM",
    ))
    model = wrap_hf_candidate(model, adapter_config).train()
    with torch.no_grad():
        initial_logits = model(examples[0], use_cache=False, logits_to_keep=1).logits
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3,
    )
    losses = []
    for tokens in islice(cycle(examples), steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite local training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        expected_logits = model(examples[0], use_cache=False, logits_to_keep=1).logits
    state = {
        f"{name}.{key}": value.detach().cpu()
        for name, module in model.named_modules() if isinstance(module, IGBundleAdapter)
        for key, value in module.state_dict().items()
    }
    torch.save(state, directory / "adapter_weights.pt")
    # Save PEFT against the original model structure for its public loader.
    # This training model is discarded after saving.
    for index, wrapper in enumerate(layers):
        layers[index] = wrapper.original_block
    model.save_pretrained(directory, safe_serialization=True)
    return adapter_config, losses, initial_logits, expected_logits


def run(checkpoint, data, output, steps, max_length):
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if max_length < 2 or steps < 1 or not checkpoint.is_dir():
        raise ValueError("Expected a checkpoint directory, positive steps and max length >= 2")
    torch.manual_seed(31)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    examples = read_examples(data, tokenizer, max_length)
    with TemporaryDirectory() as scratch:
        directory = Path(scratch)
        config, losses, initial, expected = fit_and_save(checkpoint, examples, steps, directory)
        restored = load_peft_checked(load_base(checkpoint), directory)
        restored = wrap_hf_candidate(restored, config).eval()
        tensor_count = load_weights(restored, directory / "adapter_weights.pt")
        with torch.no_grad():
            actual = restored(examples[0], use_cache=False, logits_to_keep=1).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report = {
            "steps": steps, "examples": len(examples), "losses": losses,
            "restored_adapter_tensors": tensor_count,
            "max_training_probe_logit_change": float((expected - initial).abs().max()),
            "max_reload_logit_error": float((actual - expected).abs().max()),
        }
        (directory / "igbundle_config.json").write_text(
            json.dumps({"base_model_id": str(checkpoint), "ig_adapter": vars(config)}, indent=2) + "\n",
            encoding="utf-8",
        )
        report_json = json.dumps(report, indent=2, allow_nan=False)
        (directory / "training_report.json").write_text(report_json + "\n", encoding="utf-8")
        copytree(directory, output)
    print(json.dumps(report))
    print("Training complete.")
