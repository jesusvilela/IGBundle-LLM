
import sys
import os
import torch
import numpy as np

# Add path
sys.path.append(os.path.abspath("h:/LLM-MANIFOLD/igbundle-llm/src"))

from igbundle.dynamics.fhn import FHNIntegrator
from igbundle.dynamics.train_eq_prop import EqPropTrainer

def test_associative_memory():
    print("--- Testing Equilibrium Propagation (Associative Memory) ---")
    
    # Setup
    num_fibers = 4
    # FHN nodes
    fhn = FHNIntegrator(num_fibers=num_fibers, dt=0.05)
    trainer = EqPropTrainer(fhn, beta=0.5, lr=0.1)
    
    # Task: Hebbian Association
    # Learn that Fiber 0 and Fiber 1 should be active together.
    # Target: [1, 1, 0, 0]
    # Input:  [1, 0, 0, 0] (Weak drive on 0, expecting excitation on 1 via coupling)
    
    # Training Data (We strongly drive both for learning "Togetherness")
    # Actually, for standard Hebbian, we clamp both.
    # Here, we use EqProp:
    # Input: External Drive on 0 and 1.
    input_pattern = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32) # Query
    target_pattern = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32) # Answer
    
    # To clarify: In "Training", we usually provide the full pattern as visible.
    # But here we want to learn p(y|x).
    # So we input x, and nudge y towards truth.
    
    print("Training loop...")
    losses = []
    for i in range(50):
        loss = trainer.train_step(input_pattern, target_pattern)
        losses.append(loss)
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss:.4f}")
            
    # Check Coupling Matrix
    J = fhn.coupling.data
    print("\nLearned Coupling Matrix J:")
    print(J)
    
    # Expectation: J[1,0] should be positive (0 excites 1) and J[0,1] positive (1 excites 0).
    # Indices: 0->0 (Zeroed), 0->1 ??
    # J_ij means j -> i usually? In code: matmul(J, q). i-th row = sum_j J_ij * q_j.
    # So J[1,0] is weight from 0 to 1.
    
    w_0_to_1 = J[1, 0].item()
    print(f"\nWeight 0->1: {w_0_to_1:.4f}")
    
    if w_0_to_1 > 0.05:
        print("SUCCESS: Positive coupling learned from Fiber 0 to Fiber 1.")
    else:
        print("FAILURE: Coupling too weak.")
        
    # Validation Run (Free Phase)
    print("\nValidation Run (Free Phase with Partial Input):")
    _, (q_final, _) = trainer.simulate_phase((torch.zeros(1,4), torch.zeros(1,4)), input_pattern)
    print(f"Output State: {q_final[0].tolist()}")
    
    if q_final[0, 1] > 0.5:
         print("SUCCESS: Partial input [1,0,0,0] successfully ignited Fiber 1.")
    else:
         print("FAILURE: Pattern completion failed.")

if __name__ == "__main__":
    test_associative_memory()
