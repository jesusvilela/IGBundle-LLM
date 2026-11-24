import unittest
import subprocess
import sys
import json
import os

class TestPipeline(unittest.TestCase):
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
