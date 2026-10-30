import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_gguf import DirectoryError, Reader, inspect


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def fixture(tensors=None, metadata=None, version=3):
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
    data += bytes(-len(data) % 32) + bytes(256)
    return data, prefix


class TestGGUFInspection(unittest.TestCase):
    def read_fixture(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(data)
            return inspect(path, **kwargs)

    def test_prefix_only_and_payload_mutation(self):
        data, prefix = fixture()
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
        self.assertEqual(len(report["qwen2_standard_names"]["missing"]), 14)

    def test_unsupported_version_and_magic(self):
        self.assertEqual(self.read_fixture(fixture(version=2)[0])["gguf_version"], 2)
        with self.assertRaises(DirectoryError):
            self.read_fixture(fixture(version=1)[0])
        with self.assertRaises(DirectoryError):
            self.read_fixture(b"FUGG" + fixture()[0][4:])

    def test_quantized_block_extent(self):
        data, _ = fixture(tensors=[("x", [32, 2], 8, 0)])
        self.assertEqual(self.read_fixture(data)["tensors"][0]["bytes"], 68)


if __name__ == "__main__":
    unittest.main()
