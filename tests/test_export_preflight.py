import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_gguf
from test_gguf_inspection import fixture


class ExportPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "adapter_config.json"
        self.config.write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": "example/base",
        }), encoding="utf-8")
        (self.root / "adapter_model.safetensors").write_bytes(b"fixture")
        self.converter = self.root / "convert.py"
        self.converter.write_text("# test double only\n", encoding="utf-8")
        self.output = self.root / "result"
        self.kwargs = dict(converter=self.converter, lora_only=True)

    def plan(self, **overrides):
        return export_gguf.plan_export(
            "example/base", self.root, self.output, **(self.kwargs | overrides))

    def test_explicit_scope_and_converter_are_required(self):
        for overrides in ({"lora_only": False}, {"converter": None},
                          {"converter": self.root / "missing.py"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.plan(**overrides)

    def test_config_weights_and_exact_base_are_required(self):
        for config in ({}, [], {"peft_type": "IA3"},
                       {"peft_type": "LORA", "base_model_name_or_path": "other/base"}):
            self.config.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.plan()
        self.config.write_text('{"peft_type":"LORA","base_model_name_or_path":"example/base"}',
                               encoding="utf-8")
        (self.root / "adapter_model.safetensors").rename(self.root / "unavailable")
        with self.assertRaises(FileNotFoundError):
            self.plan()

    def test_existing_destinations_are_preserved(self):
        gguf = Path(f"{self.output}.gguf")
        gguf.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.plan()
        self.assertEqual(gguf.read_bytes(), b"keep")
        gguf.rename(self.root / "existing.gguf")
        Path(f"{self.output}_hf").mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_dry_run_does_not_import_backends_or_create_outputs(self):
        with patch.dict(sys.modules, {"torch": None, "peft": None, "transformers": None}):
            plan = export_gguf.export_to_gguf_ready(
                "example/base", self.root, self.output, dry_run=True, **self.kwargs)
        self.assertFalse(plan["runtime_geometric_adapter_exported"])
        self.assertEqual(plan["base_model"], "example/base")
        self.assertFalse(Path(plan["merged_hf"]).exists())
        self.assertFalse(Path(plan["gguf"]).exists())

    def test_cli_dry_run_is_dependency_free(self):
        command = [
            sys.executable, "-S", str(Path(export_gguf.__file__).resolve()),
            "--base", "example/base", "--checkpoint", str(self.root),
            "--output", str(self.output), "--converter", str(self.converter),
            "--lora-only", "--dry-run",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["base_model"], "example/base")

    def test_execution_preflight_fails_before_backend_import(self):
        with patch.dict(sys.modules, {"torch": None, "peft": None, "transformers": None}):
            with self.assertRaisesRegex(ValueError, "lora-only"):
                export_gguf.export_to_gguf_ready(
                    "example/base", self.root, self.output, converter=self.converter)
        self.assertFalse(Path(f"{self.output}_hf").exists())

    def fake_backends(self):
        base, tokenizer, peft, merged = Mock(), Mock(), Mock(), Mock()
        peft.merge_and_unload.return_value = merged
        modules = {
            "torch": SimpleNamespace(float16="float16"),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=base)),
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            ),
            "peft": SimpleNamespace(
                PeftModel=SimpleNamespace(from_pretrained=Mock(return_value=peft)),
            ),
        }
        return base, tokenizer, peft, merged, modules

    def test_execution_uses_selected_converter_and_safe_merge(self):
        base, tokenizer, peft, merged, modules = self.fake_backends()

        def convert(command, **kwargs):
            self.assertEqual(command[1], str(self.converter))
            self.assertEqual(kwargs, {"cwd": str(self.root), "check": True})
            self.assertEqual(command[-2:], ["--outtype", "f16"])
            Path(command[command.index("--outfile") + 1]).write_bytes(fixture()[0])

        with patch.dict(sys.modules, modules), patch.object(export_gguf.subprocess, "run", side_effect=convert):
            result = export_gguf.export_to_gguf_ready(
                "example/base", self.root, self.output, **self.kwargs)
        peft.merge_and_unload.assert_called_once_with(safe_merge=True)
        merged.save_pretrained.assert_called_once_with(result["merged_hf"])
        tokenizer.save_pretrained.assert_called_once_with(result["merged_hf"])
        self.assertEqual(Path(result["gguf"]).read_bytes(), fixture()[0])
        self.assertEqual(result["directory_inspection"]["tensor_count"], 1)
        self.assertEqual(result["directory_inspection"]["tensor_payload_bytes_read"], 0)

    def test_converter_failure_propagates_and_keeps_partial_outputs(self):
        *_, modules = self.fake_backends()
        with patch.dict(sys.modules, modules):
            with patch.object(export_gguf.subprocess, "run",
                              side_effect=subprocess.CalledProcessError(1, ["converter"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    export_gguf.export_to_gguf_ready(
                        "example/base", self.root, self.output, **self.kwargs)
        self.assertTrue(Path(f"{self.output}_hf").is_dir())
        self.assertEqual(Path(f"{self.output}.gguf").read_bytes(), b"")
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_invalid_converter_output_is_not_success(self):
        *_, modules = self.fake_backends()
        with patch.dict(sys.modules, modules), patch.object(export_gguf.subprocess, "run"):
            with self.assertRaises(ValueError):
                export_gguf.export_to_gguf_ready(
                    "example/base", self.root, self.output, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
