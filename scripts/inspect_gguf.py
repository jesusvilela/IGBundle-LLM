"""Bounded GGUF v2/v3 directory inspection; never reads tensor payloads."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct


HEADER_BUDGET = 32 * 1024 * 1024
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# GGML type ID: (name, elements per block, bytes per block).
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}
SELECTED = {
    "general.architecture", "general.name", "general.file_type",
    "general.quantization_version", "general.alignment",
    "qwen2.block_count", "qwen2.embedding_length", "qwen2.context_length",
    "qwen2.feed_forward_length", "qwen2.attention.head_count",
    "qwen2.attention.head_count_kv",
}
MARKERS = re.compile(
    r"igbundle|adapter|poincare|hyperbol|hypercomplex|quaternion|octonion|"
    r"cayley|clifford|metric_chol|hamilton|sheaf|fiber", re.IGNORECASE
)


class DirectoryError(ValueError):
    pass


class Reader:
    def __init__(self, handle, budget):
        self.handle = handle
        self.budget = budget
        self.bytes_read = 0
        self.digest = hashlib.sha256()

    def read(self, size):
        if size < 0 or self.bytes_read + size > self.budget:
            raise DirectoryError("GGUF metadata/directory exceeds byte budget")
        data = self.handle.read(size)
        if len(data) != size:
            raise DirectoryError("Truncated GGUF metadata/directory")
        self.bytes_read += size
        self.digest.update(data)
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, retain=True):
        size = self.number("Q")
        if size > 1024 * 1024:
            raise DirectoryError("GGUF string exceeds 1 MiB limit")
        data = self.read(size)
        # Token strings are not published, but malformed UTF-8 still fails.
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DirectoryError("Invalid UTF-8 in GGUF string") from error
        return text if retain else None

    def value(self, kind, retain=False):
        if kind == 7:
            value = self.number("B")
            if value not in (0, 1):
                raise DirectoryError("Invalid GGUF boolean")
            return bool(value) if retain else None
        if kind in SCALARS:
            value = self.number(SCALARS[kind])
            if isinstance(value, float) and not math.isfinite(value):
                raise DirectoryError("Non-finite GGUF metadata scalar")
            return value if retain else None
        if kind == 8:
            return self.string(retain)
        if kind == 9:
            subtype, count = self.number("I"), self.number("Q")
            if subtype == 9 or subtype not in {*SCALARS, 8} or count > 1_000_000:
                raise DirectoryError("Unsupported or oversized GGUF array")
            if retain:
                raise DirectoryError("Expected scalar metadata, got array")
            for _ in range(count):
                self.value(subtype)
            return None
        raise DirectoryError(f"Unknown GGUF metadata type {kind}")


def inspect(path, budget=HEADER_BUDGET):
    path = Path(path)
    if not 0 < budget <= HEADER_BUDGET:
        raise ValueError("budget must be in (0, 32 MiB]")
    # No buffered read-ahead across the end of the tensor directory.
    with path.open("rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        reader = Reader(handle, budget)
        if reader.read(4) != b"GGUF":
            raise DirectoryError("Not a little-endian GGUF file")
        version = reader.number("I")
        if version not in (2, 3):
            raise DirectoryError(f"Unsupported GGUF version {version}")
        count, kv_count = reader.number("Q"), reader.number("Q")
        if count > 100_000 or kv_count > 4096:
            raise DirectoryError("GGUF count limit exceeded")
        metadata, keys = {}, set()
        for _ in range(kv_count):
            key = reader.string()
            if key in keys:
                raise DirectoryError(f"Duplicate metadata key: {key}")
            keys.add(key)
            value = reader.value(reader.number("I"), key in SELECTED)
            if key in SELECTED:
                metadata[key] = value
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise DirectoryError("Invalid GGUF alignment")
        tensors, names = [], set()
        for _ in range(count):
            name, dimensions = reader.string(), reader.number("I")
            if not name or name in names or not 1 <= dimensions <= 4:
                raise DirectoryError("Empty/duplicate tensor name or invalid rank")
            names.add(name)
            shape = [reader.number("Q") for _ in range(dimensions)]
            kind, offset = reader.number("I"), reader.number("Q")
            if any(d == 0 for d in shape) or offset % alignment:
                raise DirectoryError("Zero tensor dimension or unaligned offset")
            if kind not in TYPES:
                raise DirectoryError(f"Unsupported GGML tensor type {kind}; no extent certification")
            type_name, block, size = TYPES[kind]
            if shape[0] % block:
                raise DirectoryError("Tensor row is not divisible by quantization block")
            tensors.append({"name": name, "shape": shape, "type": type_name,
                            "offset": offset, "bytes": math.prod(shape) // block * size})
        directory_end = reader.bytes_read
        data_start = (directory_end + alignment - 1) // alignment * alignment
        # Padding is not read. This digest identifies a prefix, NOT model weights.
        previous_end = 0
        if data_start > before.st_size:
            raise DirectoryError("Truncated GGUF before tensor data")
        for tensor in sorted(tensors, key=lambda row: row["offset"]):
            end = tensor["offset"] + tensor["bytes"]
            if tensor["offset"] < previous_end:
                raise DirectoryError("Overlapping tensor extents")
            if data_start + end > before.st_size:
                raise DirectoryError("Tensor extent exceeds file size")
            previous_end = end
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise DirectoryError("GGUF changed during inspection")

    standard_names = None
    if metadata.get("general.architecture") == "qwen2":
        blocks = metadata.get("qwen2.block_count")
        if type(blocks) is not int or not 1 <= blocks <= 4096:
            raise DirectoryError("Invalid qwen2 block count")
        suffixes = ("attn_norm.weight", "attn_q.weight", "attn_q.bias",
                    "attn_k.weight", "attn_k.bias", "attn_v.weight", "attn_v.bias",
                    "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                    "ffn_up.weight", "ffn_down.weight")
        required = {"token_embd.weight", "output_norm.weight"} | {
            f"blk.{i}.{suffix}" for i in range(blocks) for suffix in suffixes}
        standard_names = {
            "missing": sorted(required - names),
            "extra": sorted(names - required - {"output.weight"}),
            "output_weight_present": "output.weight" in names,
            "scope": "Names only; not shape, checkpoint identity, or runtime validation",
        }
    return {
        "protocol": "gguf-directory-v1", "file_name": path.name,
        "file_bytes": before.st_size, "gguf_version": version,
        "metadata": metadata, "metadata_keys": sorted(keys),
        "tensor_count": count, "type_counts": dict(sorted(Counter(t["type"] for t in tensors).items())),
        "prefix_bytes": directory_end, "prefix_sha256": reader.digest.hexdigest(),
        "tensor_data_offset": data_start, "tensor_payload_bytes_read": 0,
        "tensor_extent_check": "within_file_nonoverlapping",
        "marker_pattern": MARKERS.pattern,
        "marker_tensor_names": sorted(name for name in names if MARKERS.search(name)),
        "marker_metadata_keys": sorted(key for key in keys if MARKERS.search(key)),
        "qwen2_standard_names": standard_names, "tensors": tensors,
        "limitations": [
            "Prefix SHA256 excludes all weights and cannot identify the full checkpoint.",
            "Names and dimensions do not establish learned geometry or hypercomplex algebra.",
            "A companion adapter requires an independently verified runtime execution path.",
            "Merged LoRA can change ordinary weights without adding tensor names.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.model)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
