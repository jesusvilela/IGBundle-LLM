import unittest
import subprocess
import sys
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

class TestPipeline(unittest.TestCase):
    def test_train_local_qwen_saves_a_strictly_reloadable_pair(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, data, output = root / "base", root / "train.jsonl", root / "result"
            Qwen2ForCausalLM(Qwen2Config(
                vocab_size=4, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2,
            )).save_pretrained(base)
            Qwen2Tokenizer(vocab={"<|endoftext|>": 0, "a": 1, "b": 2, "c": 3}).save_pretrained(base)
            data.write_text('{"text":"abc"}\n', encoding="utf-8")
            command = [
                sys.executable, "train.py", "--local-checkpoint", str(base),
                "--train-jsonl", str(data), "--output-dir", str(output), "--max_steps", "1",
            ]
            result = subprocess.run(
                command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                text=True, timeout=60, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("adapter_weights.pt", "adapter_config.json", "adapter_model.safetensors"):
                self.assertTrue((output / name).is_file(), name)
            report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["steps"], 1)
            self.assertGreater(report["restored_adapter_tensors"], 0)
            self.assertGreater(report["max_training_probe_logit_change"], 0)
            self.assertEqual(report["max_reload_logit_error"], 0)

    def test_training_smoke(self):
        """Run train.py in smoke_test mode to verify pipeline integrity."""
        # Use the same python interpreter
        cmd = [sys.executable, "train.py", "--smoke_test", "--max_steps", "1"]
        
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        
        # Run command
        result = subprocess.run(
            cmd, 
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        self.assertEqual(result.returncode, 0, f"Training Smoke Test Failed:\n{result.stderr}")
        self.assertIn("Training complete", result.stdout)
        report = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("{")))
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["steps"], 1)
        self.assertGreater(report["adapter_tensors_updated"], 0)
        self.assertGreater(report["lora_tensors_updated"], 0)
        self.assertEqual(report["frozen_tensors_updated"], 0)

if __name__ == "__main__":
    unittest.main()
