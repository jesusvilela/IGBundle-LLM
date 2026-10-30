import ast
import copy
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from igbundle.integrations.checkpoint import (
    load_weights, require_checkpoint_file, restore_adapter_state,
)
from igbundle.integrations.hf_patch import IGBundleBlockWrapper, StateCollector
from igbundle.modules.adapter import IGBundleAdapter

torch.set_num_threads(1)


def adapter():
    return IGBundleAdapter(SimpleNamespace(
        hidden_size=4, bottleneck_dim=3, num_components=2,
        num_categories=2, latent_dim=2, alpha=1., beta=1.,
        eta_f=.1, eta_b=.1, eta_w=.1, adapter_scale=.1, dropout=0.,
    ))


def model():
    return nn.Sequential(
        IGBundleBlockWrapper(nn.Linear(4, 4), adapter()),
        IGBundleBlockWrapper(nn.Linear(4, 4), adapter()),
    )


def adapter_state(m):
    return {k: v.clone() for k, v in m.state_dict().items() if ".adapter." in k}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = model()
        self.state = adapter_state(self.model)

    def assert_rejected_unchanged(self, state, **kwargs):
        before = copy.deepcopy(self.model.state_dict())
        with self.assertRaises(ValueError):
            restore_adapter_state(self.model, state, **kwargs)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def test_roundtrip_restores_every_tensor_and_not_base(self):
        state = {k: torch.full_like(v, .125) for k, v in self.state.items()}
        before = copy.deepcopy(self.model.state_dict())
        self.assertEqual(restore_adapter_state(self.model, state), len(state))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, state.get(key, before[key])), key)

    def test_missing_extra_and_wrong_prefix_rejected_before_copy(self):
        for variant in (
            {}, {k: v for k, v in self.state.items() if not k.startswith("1.")},
            {**self.state, "unexpected.weight": torch.ones(1)},
            {"base_model." + k: v for k, v in self.state.items()},
        ):
            with self.subTest(keys=len(variant)):
                self.assert_rejected_unchanged(variant)

    def test_late_bad_tensor_never_partially_mutates(self):
        key = list(self.state)[-1]
        for value in (torch.ones(123), "not a tensor", torch.tensor(1),
                      torch.full_like(self.state[key], float("nan")),
                      torch.full_like(self.state[key], float("inf"))):
            state = {k: torch.full_like(v, .5) for k, v in self.state.items()}
            state[key] = value
            self.assert_rejected_unchanged(state)

    def test_overflow_on_dtype_conversion_rejected(self):
        self.model.half()
        key = list(self.state)[-1]
        self.state[key] = torch.full_like(self.state[key], 1e30)
        self.assert_rejected_unchanged(self.state)

    def test_full_state_ignores_known_base_but_not_unknown_keys(self):
        full = copy.deepcopy(self.model.state_dict())
        full["0.original_block.weight"].fill_(77)
        before = self.model[0].original_block.weight.clone()
        restore_adapter_state(self.model, full, full_state=True)
        self.assertTrue(torch.equal(self.model[0].original_block.weight, before))
        full["unknown.adapter.weight"] = torch.ones(1)
        self.assert_rejected_unchanged(full, full_state=True)

    def test_no_registered_adapter_and_invalid_mapping(self):
        with self.assertRaisesRegex(ValueError, "No default"):
            restore_adapter_state(nn.Linear(2, 2), {})
        self.assert_rejected_unchanged([])
        self.assert_rejected_unchanged({1: torch.ones(1)})

    def test_reject_meta_sparse_complex_and_integer_adapter_values(self):
        key = next(iter(self.state))
        for value in (self.state[key].to("meta"), self.state[key].to_sparse(),
                      self.state[key].to(torch.complex64), self.state[key].long()):
            self.assert_rejected_unchanged({**self.state, key: value})

    def test_tensor_only_file_loading_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            with self.assertRaises(FileNotFoundError):
                load_weights(self.model, path)
            torch.save(self.state, path)
            with patch("torch.load", wraps=torch.load) as read:
                self.assertEqual(load_weights(self.model, path), len(self.state))
                read.assert_called_once_with(path, map_location="cpu", weights_only=True)

    def test_unsafe_pickle_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            path.touch()
            with patch("torch.load", side_effect=RuntimeError("unsupported payload")) as read:
                with self.assertRaisesRegex(RuntimeError, "unsupported payload"):
                    load_weights(self.model, path)
                self.assertEqual(read.call_count, 1)

    def test_restored_operator_executes_and_gradient_flows(self):
        wrapper = IGBundleBlockWrapper(nn.Identity(), adapter())
        state = {k: v.clone() for k, v in wrapper.state_dict().items()}
        state["adapter.out_proj.bias"].fill_(.25)
        restore_adapter_state(wrapper, state)
        collector = StateCollector()
        collector.attach(wrapper)
        hidden = torch.zeros(1, 2, 4, requires_grad=True)
        output = wrapper(hidden)
        torch.testing.assert_close(output, torch.full_like(hidden, .025))
        self.assertEqual(len(collector.states), 1)
        output.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(wrapper.adapter.out_proj.bias.grad.abs().sum().item(), 0)
        collector.detach()

    def test_five_entrypoints_reject_missing_weights_before_model_work(self):
        targets = {
            "app.py": ("load_model", ("unused.yaml",)),
            "evaluate.py": ("evaluate", ()),
            "benchmark.py": ("__init__", (SimpleNamespace(), "unused-model")),
            "bench_arena.py": ("generate_arena_answers", ("unused-model",)),
            "validate_effects.py": ("inject_adapter", (Mock(), {})),
        }
        with tempfile.TemporaryDirectory() as directory:
            for filename, (function, args) in targets.items():
                with self.subTest(filename=filename):
                    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
                    node = next(n for n in ast.walk(tree)
                                if isinstance(n, ast.FunctionDef) and n.name == function)
                    parser = Mock()
                    parser.parse_args.return_value = SimpleNamespace(checkpoint=directory)
                    namespace = {
                        "os": os, "require_checkpoint_file": require_checkpoint_file,
                        "argparse": SimpleNamespace(ArgumentParser=lambda: parser),
                    }
                    exec(compile(ast.Module(body=[node], type_ignores=[]),
                                 filename, "exec"), namespace)
                    with self.assertRaises(FileNotFoundError):
                        call_args = args if function == "evaluate" else (*args, directory)
                        namespace[function](*call_args)
                    self.assertTrue(any(
                        isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "load_weights" for n in ast.walk(node)
                    ))
                    for n in ast.walk(node):
                        if isinstance(n, ast.Call):
                            self.assertFalse(any(
                                k.arg == "strict" and isinstance(k.value, ast.Constant)
                                and k.value.value is False for k in n.keywords
                            ))


    def test_injection_propagates_restore_and_declared_peft_errors(self):
        tree = ast.parse((ROOT / "validate_effects.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "inject_adapter")
        for declared in (False, True):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "adapter_weights.pt").touch()
                if declared:
                    (Path(directory) / "adapter_config.json").write_text("{}")
                restore = Mock(side_effect=ValueError("incomplete state"))
                peft = Mock()
                peft.from_pretrained.side_effect = RuntimeError("invalid PEFT")
                m = Mock(config=SimpleNamespace(hidden_size=4))
                namespace = {
                    "os": os, "require_checkpoint_file": require_checkpoint_file,
                    "wrap_hf_candidate": lambda model, config: model,
                    "load_weights": restore,                     "load_peft_checked": peft.from_pretrained,
                }
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "validate_effects.py", "exec"), namespace)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    with self.assertRaisesRegex(
                        RuntimeError if declared else ValueError,
                        "invalid PEFT" if declared else "incomplete state",
                    ):
                        namespace["inject_adapter"](m, {"ig_adapter": {}}, directory)
                self.assertNotIn("restored", output.getvalue())
                if declared:
                    restore.assert_not_called()
                else:
                    restore.assert_called_once()


if __name__ == "__main__":
    unittest.main()
