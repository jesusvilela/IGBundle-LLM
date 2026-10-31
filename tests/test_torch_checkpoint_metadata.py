"""Hosted CPU-only comparisons against genuine, tiny torch.save state maps."""

from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_torch_checkpoint import MetadataError, inspect

torch.set_num_threads(1)


class TorchCheckpointMetadataTests(unittest.TestCase):
    def save_and_inspect(self, state):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(state, path, pickle_protocol=2)
            return inspect(path)

    def assert_layouts(self, state, report):
        self.assertEqual([entry["name"] for entry in report["tensors"]], list(state))
        stores = {entry["key"]: entry for entry in report["storages"]}
        identities = {}
        for entry in report["tensors"]:
            tensor = state[entry["name"]]
            storage = stores[entry["storage_key"]]
            self.assertEqual(entry["shape"], list(tensor.shape))
            self.assertEqual(entry["stride"], list(tensor.stride()))
            self.assertEqual(entry["storage_offset"], tensor.storage_offset())
            self.assertEqual(entry["requires_grad"], tensor.requires_grad)
            self.assertEqual(entry["dtype"], str(tensor.dtype).removeprefix("torch."))
            self.assertEqual(storage["declared_bytes"], tensor.untyped_storage().nbytes())
            self.assertEqual(storage["elements"], tensor.untyped_storage().nbytes() // tensor.element_size())
            self.assertEqual(storage["saved_device"], "cpu")
            identity = tensor.untyped_storage()._cdata
            if identity in identities:
                self.assertEqual(entry["storage_key"], identities[identity])
            else:
                self.assertNotIn(entry["storage_key"], identities.values())
                identities[identity] = entry["storage_key"]
        self.assertEqual(report["storage_count"], len(identities))
        self.assertEqual(report["sum_tensor_elements"], sum(tensor.numel() for tensor in state.values()))
        self.assertEqual(report["tensor_storage_members_opened"], 0)

    def test_actual_views_dtypes_and_aliases(self):
        base = torch.arange(24, dtype=torch.float32, requires_grad=True)
        state = OrderedDict(
            base=base, transpose=base.reshape(4, 6).T, slice=base[3:21:2],
            scalar=base[3], empty_view=base[24:24],
            empty_storage=torch.empty(0, 3),
            expanded=base[:2].expand(3, 2),
        )
        for dtype in (torch.float64, torch.float16, torch.bfloat16, torch.int64,
                      torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            state[str(dtype)] = torch.ones(2, dtype=dtype)
        self.assert_layouts(state, self.save_and_inspect(state))

    def test_actual_module_versions_and_long_memos(self):
        torch.manual_seed(31)
        module = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
        state = module.state_dict()
        report = self.save_and_inspect(state)
        self.assert_layouts(state, report)
        self.assertEqual(report["module_versions"], {key: value["version"] for key, value in state._metadata.items()})
        many = {f"tiny_{index}": torch.tensor([index], dtype=torch.int32) for index in range(300)}
        self.assert_layouts(many, self.save_and_inspect(many))

    def test_whole_modules_and_parameters_are_not_supported(self):
        for value in (torch.nn.Linear(2, 2), {"parameter": torch.nn.Parameter(torch.ones(2))}):
            with self.assertRaises(MetadataError):
                self.save_and_inspect(value)


if __name__ == "__main__":
    unittest.main()
