
import sys
import os
import torch
import unittest

# Add path
sys.path.append(os.path.abspath("h:/LLM-MANIFOLD/igbundle-llm/src"))

from igbundle.fibers.latent_store import FiberLatentStore
from igbundle.fibers.executor import FiberExecutor
from igbundle.dynamics.fhn import FHNIntegrator

class TestRetrospection(unittest.TestCase):
    def setUp(self):
        print("\n--- Setup Retrospection Test ---")
        self.num_fibers = 5
        self.store = FiberLatentStore(n_fibers=self.num_fibers, d_s=4, device='cpu')
        self.executor = FiberExecutor(self.store)
        
        # Ensure modules are loaded
        if not self.executor.retrospector:
            self.skipTest("Retrospection module not loaded.")

    def test_conservative_thought(self):
        print("Testing Conservative Thought (Valid)...")
        # 1. Initialize State
        self.executor.resonator_state = (torch.randn(self.num_fibers)*0.1, torch.randn(self.num_fibers)*0.1)
        
        # 2. Run Verify with random drive (simulating a thought)
        # Assuming FHN is somewhat reversible on short timescales
        # We might need to relax tolerance if FHN is strongly dissipative
        valid = self.executor.verify_thought([0, 1])
        
        loss, _ = self.executor.retrospector.retrospect(self.executor.resonator_state, torch.zeros(self.num_fibers))
        print(f"Conservative Loss: {loss.mean().item():.6f}")
        
        # We expect this to be largely valid or low loss
        self.assertTrue(loss.mean().item() < 1.0, "Conservative thought should have low reconstruction error")

    def test_invalid_thought_detection(self):
        print("Testing Invalid Thought (Entropy Injection)...")
        # 1. Hack the retrospector to simulate information destruction
        # We will wrap the integrator to add noise during backward pass
        original_step = self.executor.resonator.step
        
        def noisy_step(state, drive):
            q, p = original_step(state, drive)
            # Inject Chaos during integration (simulating non-conservative compute)
            return q + torch.randn_like(q) * 1.0, p
            
        # Swap method momentarily (dangerous but effective for test)
        self.executor.resonator.step = noisy_step
        
        # 2. Run Verify
        # This SHOULD fail because the backward path will diverge/hallucinate
        valid = self.executor.verify_thought([2, 3])
        
        # Restore
        self.executor.resonator.step = original_step
        
        print(f"Invalid Thought Validity: {valid}")
        self.assertFalse(valid, "Noisy integrator should fail retrospection check")

if __name__ == "__main__":
    unittest.main()
