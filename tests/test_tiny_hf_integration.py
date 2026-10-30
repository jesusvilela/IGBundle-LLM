"""Random tiny Qwen2 integration, not pretrained-checkpoint certification."""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.modules["torchao"] = None

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from igbundle.integrations.checkpoint import load_weights, load_peft_checked, verify_peft_state
from igbundle.integrations.hf_patch import IGBundleBlockWrapper, wrap_hf_candidate
from igbundle.modules.adapter import IGBundleAdapter

torch.set_num_threads(1)


def adapter_config():
    return SimpleNamespace(
        hidden_size=16, bottleneck_dim=4, num_components=2, latent_dim=2,
        num_categories=2, alpha=1., beta=1., eta_f=.1, eta_b=.1, eta_w=.1,
        adapter_scale=.1, dropout=0.,
    )


def base_model():
    torch.manual_seed(31)
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return Qwen2ForCausalLM(config).eval()


def peft_model():
    return get_peft_model(base_model(), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        lora_dropout=0., task_type="CAUSAL_LM",
    )).eval()


class TinyHFTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.tensor([[1, 4, 5, 6]])

    def test_identity_wrapper_preserves_real_qwen_logits(self):
        model = base_model()
        self.assertLess(sum(p.numel() for p in model.parameters()), 100_000)
        with torch.no_grad():
            before = model(self.tokens).logits
            wrapped = wrap_hf_candidate(model, adapter_config())
            after = wrapped(self.tokens).logits
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertEqual(sum(isinstance(m, IGBundleAdapter) for m in model.modules()), 2)

    def test_real_peft_reload_then_adapter_restore_preserves_logits_and_base(self):
        source = peft_model()
        # A separately saved PEFT checkpoint, before IGBundle wrapper insertion.
        with tempfile.TemporaryDirectory() as directory:
            for name, value in source.named_parameters():
                if "lora_B" in name:
                    with torch.no_grad():
                        value.fill_(.02)
            source.save_pretrained(directory)
            restored = load_peft_checked(base_model(), directory).eval()
            source = wrap_hf_candidate(source, adapter_config()).eval()
            restored = wrap_hf_candidate(restored, adapter_config()).eval()
            with torch.no_grad():
                initial = restored(self.tokens).logits.clone()
            with torch.no_grad():
                for module in source.modules():
                    if isinstance(module, IGBundleAdapter):
                        module.out_proj.bias.copy_(torch.linspace(-.2, .2, 16))
            state = {k: v.clone() for k, v in source.state_dict().items() if ".adapter." in k}
            before = {k: v.clone() for k, v in restored.state_dict().items() if ".adapter." not in k}
            path = Path(directory) / "adapter_weights.pt"
            torch.save(state, path)
            self.assertEqual(load_weights(restored, path), len(state))
            with torch.no_grad():
                torch.testing.assert_close(source(self.tokens).logits, restored(self.tokens).logits)
                self.assertGreater((restored(self.tokens).logits - initial).abs().max().item(), 1e-6)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_loss_backward_reaches_adapter_and_lora(self):
        model = wrap_hf_candidate(peft_model(), adapter_config()).train()
        output = model(self.tokens, labels=self.tokens, use_cache=False)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        groups = {"adapter": [], "lora": []}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if ".adapter." in name:
                    groups["adapter"].append(parameter.grad.abs().sum().item())
                if "lora_" in name:
                    groups["lora"].append(parameter.grad.abs().sum().item())
        self.assertGreater(sum(groups["adapter"]), 0)
        self.assertGreater(sum(groups["lora"]), 0)

    def test_cached_decode_matches_full_prefix_and_generation_runs(self):
        model = wrap_hf_candidate(peft_model(), adapter_config()).eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, IGBundleAdapter):
                    module.out_proj.bias.copy_(torch.linspace(-.2, .2, 16))
            full = model(self.tokens, use_cache=False).logits[:, -1]
            prefix = model(self.tokens[:, :-1], use_cache=True)
            tail = model(self.tokens[:, -1:], past_key_values=prefix.past_key_values,
                         use_cache=True).logits[:, -1]
            torch.testing.assert_close(tail, full, rtol=1e-5, atol=1e-6)
            generated = model.generate(
                self.tokens, attention_mask=torch.ones_like(self.tokens),
                max_new_tokens=2, min_new_tokens=2, do_sample=False,
            )
        self.assertEqual(generated.shape, (1, 6))
        self.assertTrue(torch.equal(generated[:, :4], self.tokens))

    def test_attention_attribute_and_cpu_dtype_are_not_host_gpu_dependent(self):
        original = torch.nn.Identity()
        original.attention_type = "declared-type"
        wrapper = IGBundleBlockWrapper(original, IGBundleAdapter(adapter_config()))
        self.assertEqual(wrapper.attention_type, "declared-type")
        fallback = IGBundleBlockWrapper(torch.nn.Identity(), IGBundleAdapter(adapter_config()))
        self.assertEqual(fallback.attention_type, "sdpa")
        with patch.object(torch.cuda, "is_bf16_supported", return_value=True) as gpu_probe:
            model = wrap_hf_candidate(base_model(), adapter_config())
        gpu_probe.assert_not_called()
        for module in model.modules():
            if isinstance(module, IGBundleAdapter):
                self.assertEqual(next(module.parameters()).dtype, torch.float32)

    def test_partial_and_extra_lora_state_fail_closed(self):
        source = peft_model()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            path = str(Path(directory) / "adapter_model.safetensors")
            state = load_file(path)
            first = next(iter(state))
            for bad in (
                {k: v for k, v in state.items() if k != first},
                {**state, "unexpected.weight": torch.ones(1)},
            ):
                save_file(bad, path)
                with self.assertRaisesRegex(ValueError, "LoRA coverage"):
                    load_peft_checked(base_model(), directory)

    def test_wrapped_lora_names_require_matching_wrapper_structure(self):
        source = wrap_hf_candidate(peft_model(), adapter_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "LoRA coverage"):
                load_peft_checked(base_model(), directory)
            restored = load_peft_checked(
                wrap_hf_candidate(base_model(), adapter_config()), directory,
            )
            expected = get_peft_model_state_dict(source)
            self.assertEqual(verify_peft_state(restored, expected), len(expected))

    def test_nonfinite_and_nonmatching_values_are_rejected(self):
        model = peft_model()
        state = get_peft_model_state_dict(model)
        first = next(iter(state))
        for value in (torch.full_like(state[first], float("nan")), state[first] + 1):
            with self.assertRaises(ValueError):
                verify_peft_state(model, {**state, first: value})

    def test_public_loader_and_export_paths_use_checked_lora(self):
        import ast
        for filename in ("app.py", "evaluate.py", "benchmark.py",
                         "bench_arena.py", "validate_effects.py", "export_gguf.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertTrue(any(isinstance(n.func, ast.Name)
                                and n.func.id == "load_peft_checked" for n in calls), filename)
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("verify_peft_checkpoint(model, checkpoint_path)", app)


if __name__ == "__main__":
    unittest.main()
