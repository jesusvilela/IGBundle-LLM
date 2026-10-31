import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_gguf import DirectoryError, Reader, inspect, qwen2_layout


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def fixture(tensors=None, metadata=None, version=3, payload_size=256):
    tensors = tensors if tensors is not None else [("example.weight", [2, 2], 0, 0)]
    metadata = metadata if metadata is not None else [
        ("general.architecture", 8, string("fixture"))]
    data = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, kind, value in metadata:
        data += string(key) + struct.pack("<I", kind) + value
    for name, shape, kind, offset in tensors:
        data += string(name) + struct.pack("<I", len(shape))
        data += b"".join(struct.pack("<Q", d) for d in shape)
        data += struct.pack("<IQ", kind, offset)
    prefix = data
    data += bytes(-len(data) % 32) + bytes(payload_size)
    return data, prefix


def qwen_fixture(*, tensor_shapes=None, metadata_values=None, omit=(), optional=False,
                 token_count=5):
    shapes = {
        "token_embd.weight": [4, 5], "output_norm.weight": [4],
        "blk.0.attn_norm.weight": [4], "blk.0.attn_q.weight": [4, 4],
        "blk.0.attn_k.weight": [4, 2], "blk.0.attn_v.weight": [4, 2],
        "blk.0.attn_output.weight": [4, 4], "blk.0.ffn_norm.weight": [4],
        "blk.0.ffn_gate.weight": [4, 6], "blk.0.ffn_up.weight": [4, 6],
        "blk.0.ffn_down.weight": [6, 4],
    }
    if optional:
        shapes.update({"output.weight": [4, 5], "output.bias": [5],
                       "blk.0.attn_q.bias": [4], "blk.0.attn_k.bias": [2],
                       "blk.0.attn_v.bias": [2]})
    shapes.update(tensor_shapes or {})
    values = {"general.architecture": "qwen2", "qwen2.block_count": 1,
              "qwen2.embedding_length": 4, "qwen2.feed_forward_length": 6,
              "qwen2.attention.head_count": 2, "qwen2.attention.head_count_kv": 1}
    values.update(metadata_values or {})
    metadata = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            metadata.append((key, 7, struct.pack("<B", value)))
        elif isinstance(value, int):
            metadata.append((key, 4, struct.pack("<I", value)))
        else:
            metadata.append((key, 8, string(value)))
    if token_count is not None:
        metadata.append(("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, token_count)
                         + b"".join(string(f"token-{i}") for i in range(token_count))))
    tensors, offset = [], 0
    for name, shape in shapes.items():
        if name not in omit:
            tensors.append((name, shape, 0, offset))
            offset += math.prod(shape) * 4
            offset += -offset % 32
    return fixture(tensors=tensors, metadata=metadata, payload_size=offset)


class TestGGUFInspection(unittest.TestCase):
    def read_fixture(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(data)
            return inspect(path, **kwargs)

    def test_prefix_only_and_payload_mutation(self):
        for factory in (fixture, qwen_fixture):
            data, prefix = factory()
            original_read = Reader.read
            def guarded_read(reader, size):
                if reader.handle.tell() + size > len(prefix):
                    raise AssertionError("Inspector attempted to read padding or tensor payload")
                return original_read(reader, size)
            with mock.patch.object(Reader, "read", guarded_read):
                report = self.read_fixture(data)
            self.assertEqual(report["prefix_sha256"], hashlib.sha256(prefix).hexdigest())
            self.assertEqual(report["prefix_bytes"], len(prefix))
            self.assertEqual(report["tensor_payload_bytes_read"], 0)
            modified = data[:-1] + b"\x7f"
            self.assertEqual(report, self.read_fixture(modified))

    def test_truncation_and_budget(self):
        data, prefix = fixture()
        for cut in (3, 10, len(prefix) - 1, len(prefix) + 1):
            with self.subTest(cut=cut), self.assertRaises(DirectoryError):
                self.read_fixture(data[:cut])
        with self.assertRaisesRegex(DirectoryError, "budget"):
            self.read_fixture(data, budget=12)

    def test_duplicate_keys_and_names(self):
        for data, _ in (
            fixture(metadata=[("x", 4, struct.pack("<I", 1))] * 2),
            fixture(tensors=[("x", [2], 0, 0), ("x", [2], 0, 32)]),
        ):
            with self.assertRaises(DirectoryError):
                self.read_fixture(data)

    def test_bad_tensor_descriptors(self):
        for tensors in (
            [("x", [0], 0, 0)], [("x", [2], 999, 0)],
            [("x", [2], 0, 1)], [("x", [2], 0, 4096)],
            [("x", [2], 8, 0)], [("x", [2] * 5, 0, 0)],
            [("x", [16], 0, 0), ("y", [16], 0, 32)],
        ):
            with self.subTest(tensors=tensors), self.assertRaises(DirectoryError):
                self.read_fixture(fixture(tensors=tensors)[0])

    def test_metadata_array_and_markers(self):
        array = struct.pack("<IQ", 8, 2) + string("one") + string("two")
        data, _ = fixture(
            tensors=[("fiber.metric_chol", [2], 0, 0)],
            metadata=[("tokenizer.ggml.tokens", 9, array)])
        report = self.read_fixture(data)
        self.assertEqual(report["metadata"], {})
        self.assertEqual(report["marker_tensor_names"], ["fiber.metric_chol"])
        self.assertIsNone(report["qwen2_standard_names"])

    def test_invalid_metadata(self):
        cases = [
            [("general.alignment", 4, struct.pack("<I", 3))],
            [("general.alignment", 7, b"\x01")],
            [("x", 9, struct.pack("<IQ", 9, 1))],
            [("x", 9, struct.pack("<IQ", 0, 1_000_001))],
            [("x", 99, b"")],
            [("x", 7, b"\x02")],
            [("x", 6, struct.pack("<f", float("nan")))],
            [("x", 8, struct.pack("<Q", 1) + b"\xff")],
        ]
        for metadata in cases:
            with self.subTest(metadata=metadata), self.assertRaises(DirectoryError):
                self.read_fixture(fixture(metadata=metadata)[0])

    def test_qwen_schema_is_names_not_geometry(self):
        data, _ = fixture(metadata=[
            ("general.architecture", 8, string("qwen2")),
            ("qwen2.block_count", 4, struct.pack("<I", 1))])
        report = self.read_fixture(data)
        self.assertEqual(report["qwen2_standard_names"]["extra"], ["example.weight"])
        self.assertEqual(len(report["qwen2_standard_names"]["missing"]), 11)
        self.assertEqual(report["qwen2_shape_check"]["status"], "failed")

    def test_unsupported_version_and_magic(self):
        self.assertEqual(self.read_fixture(fixture(version=2)[0])["gguf_version"], 2)
        with self.assertRaises(DirectoryError):
            self.read_fixture(fixture(version=1)[0])
        with self.assertRaises(DirectoryError):
            self.read_fixture(b"FUGG" + fixture()[0][4:])

    def test_quantized_block_extent(self):
        data, _ = fixture(tensors=[("x", [32, 2], 8, 0)])
        self.assertEqual(self.read_fixture(data)["tensors"][0]["bytes"], 68)

    def test_qwen_optional_biases_and_tied_output(self):
        for optional in (False, True):
            report = self.read_fixture(qwen_fixture(optional=optional)[0])
            self.assertEqual(report["qwen2_shape_check"]["status"], "passed")
            self.assertEqual(report["qwen2_shape_check"]["checked_tensors"], 16 if optional else 11)
            self.assertEqual(report["qwen2_standard_names"]["missing"], [])
            self.assertEqual(report["qwen2_standard_names"]["output_weight_present"], optional)

    def test_qwen_wrong_shapes_fail_despite_valid_names_and_extents(self):
        for name, shape in {
            "token_embd.weight": [4, 4], "blk.0.attn_q.weight": [4, 3],
            "blk.0.attn_k.weight": [4, 4], "blk.0.attn_v.bias": [4],
            "blk.0.ffn_down.weight": [4, 6], "output.bias": [4],
        }.items():
            with self.subTest(name=name):
                report = self.read_fixture(qwen_fixture(tensor_shapes={name: shape}, optional=True)[0])
                self.assertEqual(report["tensor_extent_check"], "within_file_nonoverlapping")
                self.assertEqual(report["qwen2_standard_names"]["missing"], [])
                self.assertEqual(report["qwen2_standard_names"]["extra"], [])
                check = report["qwen2_shape_check"]
                self.assertEqual(check["status"], "failed")
                self.assertEqual([row["name"] for row in check["shape_mismatches"]], [name])

    def test_qwen_inconsistent_metadata_fails(self):
        for key, value in (
            ("qwen2.embedding_length", 0), ("qwen2.feed_forward_length", True),
            ("qwen2.attention.head_count", 3), ("qwen2.attention.head_count_kv", 3),
            ("qwen2.vocab_size", 6), ("qwen2.vocab_size", "5"),
            ("qwen2.attention.key_length", 4), ("qwen2.attention.value_length", 1),
            ("qwen2.rope.dimension_count", 1),
        ):
            with self.subTest(key=key):
                report = self.read_fixture(qwen_fixture(metadata_values={key: value})[0])
                self.assertEqual(report["qwen2_shape_check"]["status"], "failed")
                self.assertTrue(report["qwen2_shape_check"]["errors"])

    def test_qwen_missing_and_extra_names_fail(self):
        for kwargs in ({"omit": ("blk.0.ffn_up.weight",)},
                       {"tensor_shapes": {"blk.1.attn_q.weight": [4, 4]}}):
            report = self.read_fixture(qwen_fixture(**kwargs)[0])
            self.assertEqual(report["qwen2_shape_check"]["status"], "failed")

    def test_qwen_vocabulary_counts_not_token_contents(self):
        report = self.read_fixture(qwen_fixture()[0])
        self.assertEqual(report["tokenizer_tokens"], {"element_type": 8, "count": 5})
        self.assertNotIn("token-0", json.dumps(report))
        for token_count in (None, 0):
            report = self.read_fixture(qwen_fixture(token_count=token_count)[0])
            self.assertEqual(report["qwen2_shape_check"]["status"], "failed")
        data, _ = fixture(metadata=[
            ("tokenizer.ggml.tokens", 8, string("not-an-array"))])
        with self.assertRaisesRegex(DirectoryError, "token array"):
            self.read_fixture(data)
        data, _ = fixture(metadata=[
            ("general.architecture", 8, string("qwen2")),
            ("qwen2.block_count", 4, struct.pack("<I", 1)),
            ("tokenizer.ggml.tokens", 9, struct.pack("<IQI", 4, 1, 0))])
        self.assertIn("tokenizer.ggml.tokens must be an array of strings",
                      self.read_fixture(data)["qwen2_shape_check"]["errors"])

    def test_qwen_native_defaults_and_explicit_vocabulary(self):
        report = self.read_fixture(qwen_fixture(
            token_count=None,
            metadata_values={"qwen2.attention.head_count_kv": None, "qwen2.vocab_size": 5},
            tensor_shapes={"blk.0.attn_k.weight": [4, 4], "blk.0.attn_v.weight": [4, 4]},
        )[0])
        self.assertEqual(report["qwen2_shape_check"]["status"], "passed")
        self.assertEqual(report["qwen2_shape_check"]["dimensions"]["kv_heads"], 2)

    def test_ambiguous_q8_1_layout_is_not_certified(self):
        with self.assertRaisesRegex(DirectoryError, "Q8_1 layout is not certified"):
            self.read_fixture(fixture(tensors=[("x", [32], 9, 0)])[0])

    def test_qwen_scalar_profile_rejects_arrays_and_invalid_blocks(self):
        for blocks in (0, 4097, True, "1"):
            with self.subTest(blocks=blocks), self.assertRaisesRegex(DirectoryError, "block count"):
                self.read_fixture(qwen_fixture(metadata_values={"qwen2.block_count": blocks})[0])
        data, _ = fixture(metadata=[
            ("qwen2.attention.head_count", 9, struct.pack("<IQI", 4, 1, 2))])
        with self.assertRaisesRegex(DirectoryError, "Expected scalar metadata"):
            self.read_fixture(data)

    def test_published_directory_shape_reports_reproduce_without_payloads(self):
        for name in ("gguf-dist-directory.json", "gguf-arc-directory.json"):
            with self.subTest(name=name):
                report = json.loads((ROOT / "docs" / "benchmark-results" / name)
                                    .read_text(encoding="utf-8"))
                names, shapes = qwen2_layout(report["metadata"], report["tensors"],
                                            report["tokenizer_tokens"])
                self.assertEqual(names, report["qwen2_standard_names"])
                self.assertEqual(shapes, report["qwen2_shape_check"])
                self.assertEqual(shapes["status"], "passed")
                self.assertEqual(shapes["checked_tensors"], report["tensor_count"])
                self.assertEqual(report["tensor_payload_bytes_read"], 0)


if __name__ == "__main__":
    unittest.main()
