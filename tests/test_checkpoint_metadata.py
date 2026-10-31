import io
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_torch_checkpoint import BudgetReader, LIMIT, MetadataError, inspect, parse_metadata


class Encoded(bytes):
    pass


def encode(value):
    if isinstance(value, Encoded):
        return bytes(value)
    if type(value) is bool:
        return b"\x88" if value else b"\x89"
    if type(value) is int:
        return b"J" + struct.pack("<i", value)
    if type(value) is str:
        data = value.encode("utf-8")
        return b"X" + struct.pack("<I", len(data)) + data
    if type(value) is tuple:
        return b"(" + b"".join(encode(item) for item in value) + b"t"
    if type(value) is dict:
        return b"}(" + b"".join(encode(key) + encode(item) for key, item in value.items()) + b"u"
    raise TypeError(type(value))


def global_ref(name):
    return Encoded(b"c" + name.replace(" ", "\n").encode("ascii") + b"\n")


def tensor(*, key="0", dtype="FloatStorage", elements=4, shape=(2, 2),
           stride=(2, 1), offset=0, gradient=False, location="cpu", hooks=None):
    storage = Encoded(encode(("storage", global_ref("torch " + dtype), key, location, elements)) + b"Q")
    if hooks is None:
        hooks = Encoded(global_ref("collections OrderedDict") + b")R")
    args = (storage, offset, shape, stride, gradient, hooks)
    return Encoded(global_ref("torch._utils _rebuild_tensor_v2") + encode(args) + b"R")


def metadata(values=None, versions=None):
    values = {"weight": tensor()} if values is None else values
    start = global_ref("collections OrderedDict") + b")R("
    body = start + b"".join(encode(key) + encode(value) for key, value in values.items()) + b"u"
    if versions is not None:
        body += encode({"_metadata": versions}) + b"b"
    return b"\x80\x02" + body + b"."


def archive_bytes(data=None, storages=None, *, compression=zipfile.ZIP_STORED,
                  metadata_name="archive/data.pkl", comment=b""):
    data = metadata() if data is None else data
    storages = {"0": 16} if storages is None else storages
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(metadata_name, data, compress_type=compression)
        for key, size in storages.items():
            archive.writestr("archive/data/" + key, bytes(size))
        archive.comment = comment
    return output.getvalue()


class CheckpointMetadataTests(unittest.TestCase):
    def inspect_bytes(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            path.write_bytes(content)
            return inspect(path)

    def test_dense_tensor_and_module_versions(self):
        report = self.inspect_bytes(archive_bytes(metadata(versions={"": {"version": 1}})))
        self.assertEqual(report["tensor_count"], 1)
        self.assertEqual(report["storage_count"], 1)
        self.assertEqual(report["sum_tensor_elements"], 4)
        self.assertEqual(report["tensors"][0]["shape"], [2, 2])
        self.assertEqual(report["tensors"][0]["stride"], [2, 1])
        self.assertEqual(report["storages"][0]["declared_bytes"], 16)
        self.assertEqual(report["module_versions"], {"": 1})
        self.assertEqual(report["tensor_storage_members_opened"], 0)

    def test_views_scalar_empty_and_saved_device(self):
        values = {
            "view": tensor(shape=(2,), stride=(1,), offset=2),
            "scalar": tensor(shape=(), stride=(), offset=1, gradient=True),
            "empty": tensor(key="1", elements=0, shape=(0, 3), stride=(3, 1)),
            "bf16": tensor(key="2", dtype="BFloat16Storage", location="cuda:0"),
            "expanded": tensor(shape=(2, 2), stride=(0, 1)),
        }
        report = self.inspect_bytes(archive_bytes(metadata(values), {"0": 16, "1": 0, "2": 8}))
        self.assertEqual(report["tensor_count"], 5)
        self.assertEqual(report["storage_count"], 3)
        self.assertEqual(report["tensors"][1]["shape"], [])
        self.assertEqual(report["tensors"][2]["shape"], [0, 3])
        self.assertEqual(report["storages"][2]["saved_device"], "cuda:0")
        self.assertEqual(report["storages"][2]["dtype"], "bfloat16")

    def test_short_and_long_memo_references(self):
        for write, read in ((b"q\x01", b"h\x01"), (b"r" + struct.pack("<I", 500), b"j" + struct.pack("<I", 500))):
            first = tensor().replace(encode("cpu"), encode("cpu") + write)
            second = tensor(key="1").replace(encode("cpu"), read)
            data = metadata({"a": Encoded(first), "b": Encoded(second)})
            state, _ = parse_metadata(data)
            self.assertEqual(state["a"].storage.location, state["b"].storage.location)

    def test_encoded_globals_are_never_called(self):
        data = metadata()
        with patch.object(pickle, "loads", side_effect=AssertionError("No unpickling")), \
             patch("collections.OrderedDict", side_effect=AssertionError("No encoded constructors")):
            state, _ = parse_metadata(data)
            self.assertEqual(state["weight"].shape, (2, 2))
        malicious = b"\x80\x02" + global_ref("builtins eval") + encode(("1 + 1",)) + b"R."
        with self.assertRaisesRegex(MetadataError, "Unsupported pickle global"):
            parse_metadata(malicious)

    def test_unsupported_or_malformed_pickle(self):
        for data in (b"", b"\x80\x04}.", metadata() + b"extra", b"\x80\x02R.",
                     b"\x80\x02h\x09.", b"\x80\x02}.", metadata({"nested": {"w": tensor()}}),
                     b"\x80\x02\x93.", metadata()[:-1]):
            with self.subTest(data=data[:20]), self.assertRaises(MetadataError):
                parse_metadata(data)

    def test_invalid_tensor_descriptions(self):
        for values in (
            {"shape": (-1, 2)}, {"shape": (2,) * 9, "stride": (1,) * 9},
            {"stride": (1,)}, {"stride": (-1, 1)}, {"offset": 2},
            {"elements": True}, {"gradient": 1}, {"key": "../0"},
            {"hooks": {"hook": 1}}, {"dtype": "UnknownStorage"},
            {"location": "../../device"},
        ):
            with self.subTest(values=values), self.assertRaises(MetadataError):
                parse_metadata(metadata({"w": tensor(**values)}))

    def test_duplicate_keys_and_unsupported_build(self):
        duplicate = b"\x80\x02}(" + (encode("w") + tensor()) * 2 + b"u."
        bad_build = metadata()[:-1] + encode({"arbitrary": 1}) + b"b."
        for data in (duplicate, bad_build, metadata(versions={"": {"version": "1"}})):
            with self.assertRaises(MetadataError):
                parse_metadata(data)

    def test_metadata_only_reads_and_payload_mutation(self):
        content = archive_bytes()
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            storage = archive.getinfo("archive/data/0")
            offset = storage.header_offset + 30 + len(storage.filename.encode()) + len(storage.extra)
        altered = bytearray(content)
        altered[offset] ^= 0xFF
        original_open = zipfile.ZipFile.open
        opened = []
        def guarded_open(archive, name, *args, **kwargs):
            filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
            opened.append(filename)
            self.assertEqual(filename, "archive/data.pkl")
            return original_open(archive, name, *args, **kwargs)
        with patch.object(zipfile.ZipFile, "open", guarded_open):
            first = self.inspect_bytes(content)
            second = self.inspect_bytes(altered)
        self.assertEqual(first, second)
        self.assertEqual(opened, ["archive/data.pkl", "archive/data.pkl"])

    def test_zip64_directory(self):
        content = archive_bytes()
        fields = struct.unpack("<4s4H2IH", content[-22:])
        offset64 = len(content) - 22
        record64 = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                               fields[4], fields[4], fields[5], fields[6])
        locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, offset64, 1)
        end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0)
        report = self.inspect_bytes(content[:-22] + record64 + locator + end)
        self.assertEqual(report["tensors"], self.inspect_bytes(content)["tensors"])

    def test_deflated_metadata_and_corruption(self):
        content = archive_bytes(compression=zipfile.ZIP_DEFLATED)
        self.assertEqual(self.inspect_bytes(content)["tensors"], self.inspect_bytes(archive_bytes())["tensors"])
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            info = archive.getinfo("archive/data.pkl")
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        corrupted = bytearray(content)
        corrupted[offset] = 7  # Invalid DEFLATE block type, before the pickle decoder.
        with self.assertRaisesRegex(MetadataError, "metadata ZIP"):
            self.inspect_bytes(corrupted)
        stored = bytearray(archive_bytes())
        stored[offset] ^= 0xFF
        with self.assertRaisesRegex(MetadataError, "CRC"):
            self.inspect_bytes(stored)

    def test_local_header_and_overlap_rejections(self):
        oversized_header = bytearray(archive_bytes())
        struct.pack_into("<H", oversized_header, 28, 4096)
        overlap = bytearray(archive_bytes())
        central = struct.unpack("<I", overlap[-6:-2])[0]
        compressed_size = struct.unpack_from("<I", overlap, central + 20)[0]
        struct.pack_into("<I", overlap, central + 20, compressed_size + 1)
        for content in (oversized_header, overlap, archive_bytes()[:-1]):
            with self.assertRaises(MetadataError):
                self.inspect_bytes(content)

    def test_missing_extra_or_inconsistent_storage(self):
        for content in (archive_bytes(storages={}), archive_bytes(storages={"0": 12}),
                        archive_bytes(storages={"0": 16, "1": 4}),
                        archive_bytes(metadata({"a": tensor(), "b": tensor(elements=5)}))):
            with self.assertRaises(MetadataError):
                self.inspect_bytes(content)

    def test_zip_bounds_and_profile_rejections(self):
        oversized_directory = bytearray(archive_bytes())
        struct.pack_into("<I", oversized_directory, len(oversized_directory) - 10, LIMIT + 1)
        for content in (b"not a zip", archive_bytes(comment=b"comment"), oversized_directory,
                        archive_bytes(compression=zipfile.ZIP_BZIP2),
                        archive_bytes(metadata_name="../data.pkl"),
                        archive_bytes(metadata_name="archive/not-metadata.pkl"),
                        archive_bytes(data=b"x" * (LIMIT + 1), compression=zipfile.ZIP_DEFLATED)):
            with self.subTest(size=len(content)), self.assertRaises(MetadataError):
                self.inspect_bytes(content)

    def test_duplicate_zip_members_and_encryption(self):
        output = io.BytesIO(archive_bytes())
        with zipfile.ZipFile(output, "a") as archive:
            with self.assertWarns(UserWarning):
                archive.writestr("archive/data.pkl", metadata())
        with self.assertRaisesRegex(MetadataError, "Duplicate ZIP"):
            self.inspect_bytes(output.getvalue())
        encrypted = bytearray(archive_bytes())
        central = struct.unpack("<I", encrypted[-6:-2])[0]
        struct.pack_into("<H", encrypted, central + 8, 1)
        with self.assertRaisesRegex(MetadataError, "encryption"):
            self.inspect_bytes(encrypted)

    def test_mutation_during_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            path.write_bytes(archive_bytes())
            before = path.stat()
            original_read = BudgetReader.read
            changed = False
            def mutate(reader, size=-1):
                nonlocal changed
                value = original_read(reader, size)
                if not changed:
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10_000_000))
                    changed = True
                return value
            with patch.object(BudgetReader, "read", mutate), self.assertRaisesRegex(MetadataError, "changed"):
                inspect(path)

    def test_read_budgets_and_eof_read(self):
        reader = BudgetReader(io.BytesIO(bytes(100)), 100)
        reader.seek(-22, 2)
        self.assertEqual(len(reader.read()), 22)
        reader.seek(0)
        with self.assertRaisesRegex(MetadataError, "approved metadata"):
            reader.read(1)
        with self.assertRaisesRegex(MetadataError, "budget"):
            reader.read(2 * LIMIT + 8193)
        with self.assertRaises(MetadataError):
            parse_metadata(bytes(LIMIT + 1))
        with self.assertRaises(MetadataError):
            parse_metadata(b"\x80\x02" + b")" * 4098)
        with self.assertRaises(MetadataError):
            parse_metadata(b"\x80\x02" + encode("x" * 4097) + b".")

    def test_cli_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("state.pt", "report.json"))
            source.write_bytes(archive_bytes())
            target.write_text("preserve", encoding="utf-8")
            result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / "inspect_torch_checkpoint.py"),
                                     str(source), "--output", str(target)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(target.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
