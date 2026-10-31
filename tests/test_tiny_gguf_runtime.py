"""Synthetic F32 Qwen2 execution, not archived-weight or adapter certification."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("DISABLE_TORCHAO", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
sys.modules["torchao"] = None

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from scripts.inspect_gguf import inspect
from test_gguf_inspection import fixture, string

NATIVE_REVISION = "067b8d7af3f4775b1ad39470d1860696a8a7b3a3"
RTOL, ATOL = 1e-4, 1e-5
torch.set_num_threads(1)


def make_model(*, tied=False):
    torch.manual_seed(23)
    config = Qwen2Config(
        vocab_size=259, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1 if tied else 2, num_attention_heads=4,
        num_key_value_heads=1 if tied else 2, max_position_embeddings=32,
        rope_theta=10000., rms_norm_eps=1e-6, attention_dropout=0.,
        tie_word_embeddings=tied, bos_token_id=256, eos_token_id=257, pad_token_id=258,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".bias"):
                parameter.copy_(torch.linspace(-.03, .03, parameter.numel()).reshape_as(parameter))
    return model


def gguf_bytes(model, *, bad_shape=None, negate_output=False):
    config = model.config
    tensors = {
        "token_embd.weight": model.model.embed_tokens.weight,
        "output_norm.weight": model.model.norm.weight,
    }
    if not config.tie_word_embeddings:
        tensors["output.weight"] = -model.lm_head.weight if negate_output else model.lm_head.weight
    for i, layer in enumerate(model.model.layers):
        prefix = f"blk.{i}."
        tensors[prefix + "attn_norm.weight"] = layer.input_layernorm.weight
        tensors[prefix + "ffn_norm.weight"] = layer.post_attention_layernorm.weight
        for name, projection in (("attn_q", layer.self_attn.q_proj),
                                 ("attn_k", layer.self_attn.k_proj),
                                 ("attn_v", layer.self_attn.v_proj),
                                 ("attn_output", layer.self_attn.o_proj),
                                 ("ffn_gate", layer.mlp.gate_proj),
                                 ("ffn_up", layer.mlp.up_proj),
                                 ("ffn_down", layer.mlp.down_proj)):
            tensors[prefix + name + ".weight"] = projection.weight
            if projection.bias is not None:
                tensors[prefix + name + ".bias"] = projection.bias
    if bad_shape:
        tensors[bad_shape] = tensors[bad_shape][:-1]

    values = {
        "general.architecture": "qwen2",
        "general.name": "synthetic-igbundle-native-check",
        "qwen2.context_length": 32,
        "qwen2.block_count": config.num_hidden_layers,
        "qwen2.embedding_length": 32,
        "qwen2.feed_forward_length": 64,
        "qwen2.attention.head_count": 4,
        "qwen2.attention.head_count_kv": config.num_key_value_heads,
        "qwen2.attention.layer_norm_rms_epsilon": config.rms_norm_eps,
        "qwen2.rope.dimension_count": 8,
        "qwen2.rope.freq_base": config.rope_theta,
        "qwen2.vocab_size": 259,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.bos_token_id": 256,
        "tokenizer.ggml.eos_token_id": 257,
        "tokenizer.ggml.padding_token_id": 258,
    }
    metadata = []
    for key, value in values.items():
        if isinstance(value, str):
            kind, encoded = 8, string(value)
        elif isinstance(value, int):
            kind, encoded = 4, struct.pack("<I", value)
        else:
            kind, encoded = 6, struct.pack("<f", value)
        metadata.append((key, kind, encoded))
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    alphabet, extra = [], 256
    for byte in range(256):
        alphabet.append(chr(byte) if byte in visible else chr(extra))
        extra += byte not in visible
    tokens = alphabet + ["<|endoftext|>", "<|im_end|>", "<|padding|>"]
    metadata.extend([
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, len(tokens))
         + b"".join(string(token) for token in tokens)),
        ("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, len(tokens))
         + struct.pack("<259i", *([1] * 256 + [3] * 3))),
        ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 0)),
    ])
    descriptors, payload = [], bytearray()
    for name, tensor in tensors.items():
        data = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        descriptors.append((name, list(reversed(data.shape)), 0, len(payload)))
        payload.extend(data.tobytes())
        payload.extend(bytes(-len(payload) % 32))
    header, _ = fixture(tensors=descriptors, metadata=metadata, payload_size=0)
    result = header + payload
    if len(result) >= 1_000_000:
        raise ValueError("Synthetic native fixture exceeds its size budget")
    return result


class TinyGGUFRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("IGBUNDLE_NATIVE_QWEN2_BINARY")
        if not binary or not Path(binary).is_file():
            raise RuntimeError("Build tests/native at the pinned revision and set IGBUNDLE_NATIVE_QWEN2_BINARY")
        cls.binary = str(Path(binary).resolve())

    def run_native(self, path, tokens, mode="full"):
        return subprocess.run([self.binary, str(path), mode, *map(str, tokens)],
                              capture_output=True, text=True, timeout=30, check=False)

    def native_logits(self, path, tokens, mode="full"):
        result = self.run_native(path, tokens, mode)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["native_revision"], NATIVE_REVISION)
        self.assertEqual(record["mode"], mode)
        self.assertEqual(record["tokens"], tokens)
        self.assertEqual(record["n_vocab"], 259)
        logits = torch.tensor(record["logits"], dtype=torch.float32)
        self.assertEqual(logits.shape, (len(tokens), 259))
        self.assertTrue(torch.isfinite(logits).all())
        return logits

    def test_native_full_and_incremental_logits(self):
        for tied in (False, True):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as directory:
                model = make_model(tied=tied)
                parameters = sum(p.numel() for p in model.parameters())
                self.assertLess(parameters, 100_000)
                path = Path(directory) / "tiny.gguf"
                path.write_bytes(gguf_bytes(model))
                report = inspect(path)
                self.assertEqual(report["qwen2_shape_check"]["status"], "passed")
                self.assertEqual(report["tensor_payload_bytes_read"], 0)
                self.assertEqual(report["qwen2_standard_names"]["output_weight_present"], not tied)
                for tokens in ([0, 1, 2, 5], [11, 9, 8, 7, 6, 4]):
                    with torch.no_grad():
                        reference = model(torch.tensor([tokens])).logits[0]
                    full = self.native_logits(path, tokens)
                    incremental = self.native_logits(path, tokens, "incremental")
                    for actual in (full, incremental):
                        torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL)
                        self.assertEqual(actual.argmax(-1).tolist(), reference.argmax(-1).tolist())
                    torch.testing.assert_close(incremental, full, rtol=RTOL, atol=ATOL)
                    print("NATIVE_GGUF_RESULT " + json.dumps({
                        "tied_output": tied, "parameters": parameters, "file_bytes": path.stat().st_size,
                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "directory_prefix_sha256": report["prefix_sha256"],
                        "tokens": tokens, "logit_shape": list(full.shape),
                        "full_max_abs_error": (full - reference).abs().max().item(),
                        "incremental_max_abs_error": (incremental - reference).abs().max().item(),
                        "batch_max_abs_error": (full - incremental).abs().max().item(),
                        "argmax_ids": reference.argmax(-1).tolist(),
                    }))

    def test_native_rejects_wrong_declared_shapes(self):
        model = make_model()
        for tensor in ("blk.0.attn_q.weight", "blk.0.attn_k.weight"):
            with self.subTest(tensor=tensor), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad-shape.gguf"
                path.write_bytes(gguf_bytes(model, bad_shape=tensor))
                report = inspect(path)
                self.assertEqual(report["qwen2_shape_check"]["status"], "failed")
                self.assertEqual(report["tensor_extent_check"], "within_file_nonoverlapping")
                result = self.run_native(path, [0, 1])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(f"tensor '{tensor}' has wrong shape", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_directory_hash_does_not_bind_executed_weights(self):
        model = make_model()
        tokens = [0, 1, 2, 5]
        with tempfile.TemporaryDirectory() as directory:
            original, modified = (Path(directory) / name for name in ("original.gguf", "modified.gguf"))
            original.write_bytes(gguf_bytes(model))
            modified.write_bytes(gguf_bytes(model, negate_output=True))
            reports = [inspect(path) for path in (original, modified)]
            for report in reports:
                report.pop("file_name")
            self.assertEqual(*reports)
            first, second = self.native_logits(original, tokens), self.native_logits(modified, tokens)
            torch.testing.assert_close(second, -first, rtol=RTOL, atol=ATOL)
            effect = (second - first).abs().max().item()
            self.assertGreater(effect, 1e-3)
            print("NATIVE_GGUF_PAYLOAD_COUNTEREXAMPLE " + json.dumps({
                "same_directory_report_except_filename": True, "output_weight_negated": True,
                "directory_prefix_sha256": reports[0]["prefix_sha256"],
                "original_file_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                "modified_file_sha256": hashlib.sha256(modified.read_bytes()).hexdigest(),
                "max_abs_logit_change": effect,
                "negation_max_abs_error": (second + first).abs().max().item(),
            }))


if __name__ == "__main__":
    unittest.main()
