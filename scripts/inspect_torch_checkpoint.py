"""Inspect bounded torch.save state-dict metadata without unpickling tensors.

Only a restricted protocol-2 tensor-map profile is supported. Pickle-declared
globals are symbolic tags, never imported or called. Storage members are not read.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickletools
import re
import struct
import zipfile
import zlib

LIMIT = 1024 * 1024
MAX_ITEMS = 4096
STORAGE_TYPES = {
    "torch FloatStorage": ("float32", 4),
    "torch DoubleStorage": ("float64", 8),
    "torch HalfStorage": ("float16", 2),
    "torch BFloat16Storage": ("bfloat16", 2),
    "torch LongStorage": ("int64", 8),
    "torch IntStorage": ("int32", 4),
    "torch ShortStorage": ("int16", 2),
    "torch CharStorage": ("int8", 1),
    "torch ByteStorage": ("uint8", 1),
    "torch BoolStorage": ("bool", 1),
}
GLOBALS = set(STORAGE_TYPES) | {"collections OrderedDict", "torch._utils _rebuild_tensor_v2"}
MARK = object()


class MetadataError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MetadataError(message)


class BudgetReader:
    def __init__(self, handle, size):
        self.handle, self.size, self.used = handle, size, 0
        self.ranges = [(max(0, size - 98), size)]
        self.directory = None
        self.directory_sha256 = None

    def read(self, size=-1):
        start = self.tell()
        if size < 0:
            size = self.size - start
        require(0 <= size and self.used + size <= 2 * LIMIT + 8192, "Metadata read budget exceeded")
        require(any(low <= start <= start + size <= high for low, high in self.ranges),
                "Read outside approved metadata regions")
        data = self.handle.read(size)
        self.used += len(data)
        require(len(data) == size, "Truncated metadata")
        if self.directory == (start, size):
            self.directory_sha256 = hashlib.sha256(data).hexdigest()
        return data

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "Invalid seek")
        position = offset + (self.tell() if whence == 1 else self.size if whence == 2 else 0)
        require(0 <= position <= self.size, "Seek outside checkpoint")
        return self.handle.seek(position)

    def tell(self):
        return self.handle.tell()

    def seekable(self):
        return True


def directory_bounds(reader):
    require(reader.size >= 98, "Not a supported torch.save ZIP archive")
    reader.seek(-22, 2)
    sig, disk, cd_disk, disk_count, count, size, offset, comment = struct.unpack("<4s4H2IH", reader.read(22))
    require(sig == b"PK\x05\x06" and not comment and not disk and not cd_disk and disk_count == count,
            "Only comment-free, single-disk ZIP archives are supported")
    end = reader.size - 22
    reader.seek(-42, 2)
    locator = reader.read(20)
    if locator[:4] == b"PK\x06\x07":
        _, disk64, offset64, disks = struct.unpack("<4sIQI", locator)
        require(disk64 == 0 and disks == 1 and offset64 == reader.size - 98, "Unsupported ZIP64 locator")
        reader.seek(offset64)
        fields = struct.unpack("<4sQ2H2I4Q", reader.read(56))
        require(fields[0] == b"PK\x06\x06" and fields[1] == 44 and fields[4] == fields[5] == 0
                and fields[6] == fields[7], "Unsupported ZIP64 record")
        count, size, offset = fields[7:10]
        end = offset64
    require(0 < count <= MAX_ITEMS and 46 <= size <= LIMIT and offset >= 0 and offset + size == end,
            "ZIP directory exceeds supported bounds")
    reader.ranges.append((offset, reader.size))
    reader.directory = (offset, size)
    return count, size, offset


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    dtype: str
    itemsize: int
    elements: int
    location: str


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple
    requires_grad: bool


class StateMap(dict):
    versions = None


def parse_metadata(data):
    require(len(data) <= LIMIT, "Pickle metadata exceeds budget")
    stack, memo, symbols = [], {}, set()

    def pop():
        require(bool(stack), "Pickle stack underflow")
        return stack.pop()

    def marked():
        values = []
        while stack and stack[-1] is not MARK:
            values.append(pop())
        require(bool(stack), "Missing pickle MARK")
        pop()
        return list(reversed(values))

    def assign(target, values):
        require(isinstance(target, dict) and len(values) % 2 == 0, "Invalid dictionary operation")
        for key, value in zip(values[::2], values[1::2]):
            require(type(key) is str and key not in target, "Duplicate or non-string dictionary key")
            require(len(target) < MAX_ITEMS, "Dictionary item budget exceeded")
            target[key] = value

    try:
        for index, (opcode, arg, position) in enumerate(pickletools.genops(data)):
            name = opcode.name
            require(index < 50000 and len(stack) <= MAX_ITEMS, "Pickle operation/stack budget exceeded")
            if index == 0:
                require(name == "PROTO" and arg == 2, "Only pickle protocol 2 is supported")
                continue
            if name in ("BINUNICODE", "SHORT_BINSTRING"):
                require(type(arg) is str and len(arg) <= MAX_ITEMS, "String exceeds metadata profile")
                stack.append(arg)
            elif name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4"):
                require(type(arg) is int and -(2**63) <= arg < 2**63, "Integer exceeds metadata profile")
                stack.append(arg)
            elif name in ("NEWTRUE", "NEWFALSE"):
                stack.append(name == "NEWTRUE")
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "MARK":
                stack.append(MARK)
            elif name == "TUPLE":
                stack.append(tuple(marked()))
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                count = int(name[-1])
                require(len(stack) >= count, "Pickle tuple underflow")
                value = tuple(stack[-count:])
                del stack[-count:]
                stack.append(value)
            elif name in ("BINPUT", "LONG_BINPUT"):
                require(bool(stack) and 0 <= arg < 16384 and arg not in memo, "Invalid pickle memo write")
                memo[arg] = stack[-1]
            elif name in ("BINGET", "LONG_BINGET"):
                require(arg in memo, "Missing pickle memo entry")
                stack.append(memo[arg])
            elif name == "GLOBAL":
                require(arg in GLOBALS, "Unsupported pickle global")
                symbols.add(arg)
                stack.append(Symbol(arg))
            elif name == "BINPERSID":
                value = pop()
                require(type(value) is tuple and len(value) == 5 and value[0] == "storage",
                        "Unsupported persistent reference")
                _, kind, key, location, elements = value
                require(isinstance(kind, Symbol) and kind.name in STORAGE_TYPES, "Unsupported storage type")
                require(type(key) is str and re.fullmatch(r"[0-9]{1,64}", key), "Invalid storage key")
                require(type(location) is str and re.fullmatch(r"[A-Za-z0-9_:]{1,64}", location),
                        "Invalid saved device label")
                require(type(elements) is int and 0 <= elements < 2**63, "Invalid storage size")
                dtype, itemsize = STORAGE_TYPES[kind.name]
                stack.append(Storage(key, dtype, itemsize, elements, location))
            elif name == "REDUCE":
                args, function = pop(), pop()
                require(isinstance(function, Symbol) and type(args) is tuple, "Invalid symbolic reduction")
                if function.name == "collections OrderedDict":
                    require(args == (), "Only empty OrderedDict construction is supported")
                    stack.append(StateMap())
                elif function.name == "torch._utils _rebuild_tensor_v2":
                    require(len(args) == 6, "Unsupported tensor rebuild signature")
                    storage, offset, shape, stride, gradient, hooks = args
                    require(isinstance(storage, Storage) and type(offset) is int and offset >= 0,
                            "Invalid tensor storage/offset")
                    require(type(shape) is tuple and type(stride) is tuple and len(shape) == len(stride)
                            and len(shape) <= 8, "Invalid tensor rank/stride")
                    require(all(type(n) is int and 0 <= n < 2**31 for n in shape)
                            and all(type(n) is int and 0 <= n < 2**63 for n in stride), "Invalid tensor dimensions")
                    require(type(gradient) is bool and isinstance(hooks, dict) and not hooks, "Unsupported tensor hooks")
                    span = 1 + sum((size - 1) * step for size, step in zip(shape, stride)) if math.prod(shape) else 0
                    require(offset + span <= storage.elements, "Tensor view exceeds declared storage")
                    stack.append(Tensor(storage, offset, shape, stride, gradient))
                else:
                    raise MetadataError("Global is not an allowed metadata constructor")
            elif name == "SETITEMS":
                values = marked()
                require(bool(stack), "Missing dictionary")
                assign(stack[-1], values)
            elif name == "SETITEM":
                value, key = pop(), pop()
                require(bool(stack), "Missing dictionary")
                assign(stack[-1], [key, value])
            elif name == "BUILD":
                state = pop()
                require(bool(stack) and isinstance(stack[-1], StateMap) and type(state) is dict
                        and set(state) == {"_metadata"}, "Unsupported pickle BUILD state")
                versions = state["_metadata"]
                require(isinstance(versions, dict), "Invalid module-version metadata")
                require(all(isinstance(value, dict) and set(value) == {"version"}
                            and type(value["version"]) is int and value["version"] >= 0
                            for value in versions.values()), "Unsupported module-version metadata")
                require(stack[-1].versions is None, "Repeated BUILD state")
                stack[-1].versions = {key: value["version"] for key, value in versions.items()}
            elif name == "STOP":
                require(position + 1 == len(data) and len(stack) == 1, "Trailing or incomplete pickle data")
                state = pop()
                require(isinstance(state, dict) and state and all(type(key) is str and key
                        and isinstance(value, Tensor) for key, value in state.items()), "Expected a flat tensor state map")
                return state, sorted(symbols)
            else:
                raise MetadataError(f"Unsupported pickle opcode: {name}")
    except (ValueError, IndexError, struct.error, UnicodeError, OverflowError) as exc:
        if isinstance(exc, MetadataError):
            raise
        raise MetadataError("Malformed pickle metadata") from exc
    raise MetadataError("Missing pickle STOP")


def inspect(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as handle:
        reader = BudgetReader(handle, before.st_size)
        count, directory_size, directory_offset = directory_bounds(reader)
        try:
            with zipfile.ZipFile(reader) as archive:
                entries = archive.infolist()
                require(len(entries) == count and len({e.filename for e in entries}) == count, "Duplicate ZIP member")
                require(len({e.header_offset for e in entries}) == count
                        and all(0 <= e.header_offset < directory_offset for e in entries), "Invalid ZIP member offsets")
                candidates = [e for e in entries if e.filename == "data.pkl" or e.filename.endswith("/data.pkl")]
                require(len(candidates) == 1, "Expected exactly one data.pkl member")
                info = candidates[0]
                parts = info.filename.split("/")
                require(len(parts) <= 2 and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                        and part not in (".", "..") for part in parts), "Unsupported metadata member path")
                require(info.file_size <= LIMIT and info.compress_size <= LIMIT, "Pickle metadata exceeds budget")
                require(not info.flag_bits & 1 and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                        "Unsupported metadata compression/encryption")
                reader.ranges.append((info.header_offset, info.header_offset + 30))
                reader.seek(info.header_offset)
                header = reader.read(30)
                require(header[:4] == b"PK\x03\x04", "Invalid metadata local header")
                name_size, extra_size = struct.unpack("<HH", header[26:])
                header_size = 30 + name_size + extra_size
                require(header_size <= 4096, "Metadata local header exceeds budget")
                end = info.header_offset + header_size + info.compress_size
                next_header = min([e.header_offset for e in entries if e.header_offset > info.header_offset]
                                  + [directory_offset])
                require(end <= next_header, "Metadata member overlaps another member")
                reader.ranges.append((info.header_offset, end))
                with archive.open(info) as member:
                    data = member.read(LIMIT + 1)
                require(len(data) == info.file_size and len(data) <= LIMIT, "Invalid pickle metadata length")
                state, symbols = parse_metadata(data)
                members = {entry.filename: entry for entry in entries}
                prefix = info.filename[:-len("data.pkl")]
                storages = {}
                tensors = []
                for name, tensor in state.items():
                    storage = tensor.storage
                    require(storage.key not in storages or storages[storage.key] == storage,
                            "Conflicting shared-storage descriptions")
                    storages[storage.key] = storage
                    member_name = prefix + "data/" + storage.key
                    require(member_name in members, "Missing declared tensor storage member")
                    entry = members[member_name]
                    require(entry.file_size == storage.elements * storage.itemsize, "Declared storage byte count mismatch")
                    tensors.append({"name": name, "dtype": storage.dtype, "shape": list(tensor.shape),
                                    "stride": list(tensor.stride), "storage_key": storage.key,
                                    "storage_offset": tensor.offset, "requires_grad": tensor.requires_grad})
                storage_members = {e.filename for e in entries if e.filename.startswith(prefix + "data/")}
                require(storage_members == {prefix + "data/" + key for key in storages}, "Unreferenced storage member")
        except (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError) as exc:
            raise MetadataError(f"Invalid or unsupported metadata ZIP: {exc}") from exc
        opened_after = os.fstat(handle.fileno())
    after = path.stat()
    signature = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
    require(signature(before) == signature(opened_after) == signature(after), "Checkpoint changed during inspection")
    require(reader.directory_sha256 is not None, "Directory fingerprint was not captured")
    return {
        "protocol": "torch-state-metadata-v1",
        "file_name": path.name, "file_bytes": before.st_size,
        "zip_members": count, "zip_directory_bytes": directory_size,
        "zip_directory_sha256": reader.directory_sha256,
        "metadata_member": info.filename, "metadata_bytes": len(data),
        "metadata_sha256": hashlib.sha256(data).hexdigest(),
        "metadata_read_bytes": reader.used, "tensor_storage_members_opened": 0,
        "pickle_globals": symbols, "tensor_count": len(tensors), "storage_count": len(storages),
        "sum_tensor_elements": sum(math.prod(tensor.shape) for tensor in state.values()),
        "tensors": tensors,
        "storages": [{"key": storage.key, "dtype": storage.dtype, "elements": storage.elements,
                      "declared_bytes": storage.elements * storage.itemsize, "saved_device": storage.location}
                     for storage in storages.values()],
        "module_versions": state.versions if isinstance(state, StateMap) else None,
        "limitations": [
            "Only a restricted protocol-2 flat tensor-map metadata profile is supported.",
            "Pickle-declared globals are symbolic; no encoded callable or torch loader is executed.",
            "No tensor storage values or payload checksums are read or verified.",
            "Metadata fingerprints do not identify the complete checkpoint.",
            "Shapes and saved flags do not certify restoration, training history or geometric laws.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = json.dumps(inspect(args.checkpoint), indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(result)
        else:
            print(result, end="")
    except (MetadataError, OSError) as exc:
        parser.exit(2, f"Checkpoint metadata inspection failed: {exc}\n")


if __name__ == "__main__":
    main()
