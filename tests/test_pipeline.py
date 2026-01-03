import unittest
import subprocess
import sys
import os

class TestPipeline(unittest.TestCase):
    def test_training_smoke(self):
        """Run train.py in smoke_test mode to verify pipeline integrity."""
        # Use the same python interpreter
        cmd = [sys.executable, "train.py", "--smoke_test", "--max_steps", "1"]
        
        # Enforce current environment
        env = os.environ.copy()
        
        # Run command
        result = subprocess.run(
            cmd, 
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, f"Training Smoke Test Failed:\n{result.stderr}")
        self.assertIn("Training complete", result.stdout)

if __name__ == "__main__":
    unittest.main()
