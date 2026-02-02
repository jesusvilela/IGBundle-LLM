
import sys
import os
import torch
import numpy as np

# Add path
sys.path.append(os.path.abspath("h:/LLM-MANIFOLD/igbundle-llm/src"))

from igbundle.fibers.latent_store import FiberLatentStore
from igbundle.fibers.executor import FiberExecutor

def test_eq_prop_logic_gate():
    print("--- Testing EqProp Logic Gate Learning (AND Gate) ---")
    
    # 0. Setup
    num_fibers = 3
    # Concept: Fiber 0 = Input A, Fiber 1 = Input B, Fiber 2 = Output (A AND B)
    
    # Init Store & Executor
    store = FiberLatentStore(n_fibers=num_fibers, d_s=4, device='cpu')
    executor = FiberExecutor(store)
    
    if not executor.eq_propagator:
        print("SKIP: EqProp not available.")
        return

    # Tune Hyperparams for Test
    executor.eq_propagator.lr = 0.1 # Boost LR
    executor.eq_propagator.beta = 0.5 # Default is 0.5


    # 1. Define Data (AND Logic)
    # Inputs: (0,0), (0,1), (1,0), (1,1)
    # Output: 0, 0, 0, 1
    # We map "0" to weak activation (0.0) and "1" to strong (1.0).
    
    dataset = [
        ([0, 1], [], [0.0, 0.0, 0.0]),       # Input: None active -> Output: 0
        ([0], [0], [1.0, 0.0, 0.0]),         # Input: A -> Output: 0 (Target 0 is implicit/active indices empty)
        ([1], [1], [0.0, 1.0, 0.0]),         # Input: B -> Output: 0
        ([0, 1], [0, 1, 2], [1.0, 1.0, 1.0]) # Input: A,B -> Output: 1 (Target includes 2)
    ]
    
    # Explicit Indices for convenience
    # format: (input_indices, target_indices_for_output_node)
    # Case 1: In [], Out [0] (Low). Target indices = []
    # Case 2: In [0], Out [0] (Low). Target indices = [0] (Input clamped high, output unclamped/low?)
    #   Wait, EqProp target clamping means we clamp the OUTPUT node.
    #   If we want Output=0, we clamp Fiber 2 to 0.0.
    #   If we want Output=1, we clamp Fiber 2 to 1.0.
    
    # Better Data Structure for `learn_step`:
    # It takes `input_indices` (Driven High) and `target_indices` (Driven High).
    # To drive something LOW, we need to modify the underlying propagator to accept a value map, 
    # or rely on "absence of drive" = 0.
    
    training_data = [
        ([], []),            # 0 AND 0 = 0 (No drive on 0,1. No drive on 2)
        ([0], [0]),          # 1 AND 0 = 0 (Drive 0. No drive on 2)
        ([1], [1]),          # 0 AND 1 = 0 (Drive 1. No drive on 2)
        ([0, 1], [0, 1, 2])  # 1 AND 1 = 1 (Drive 0,1. Drive 2)
    ]
    
    print("Training...")
    for epoch in range(50):
        total_loss = 0
        for inp_idx, tgt_idx in training_data:
            loss = executor.learn_step(inp_idx, tgt_idx)
            total_loss += loss
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
            
    # Verification
    print("\nVerification (Free Phase):")
    # Reset State for clean test
    executor.resonator_state = (torch.zeros(num_fibers), torch.zeros(num_fibers))
    
    # Test 1 AND 1 Case
    # Run loop to allow dynamics to evolve
    print("Simulating 100 steps...")
    for _ in range(100):
        executor.update_resonance([0, 1])
        
    q_out = executor.resonator_state[0] # q

    
    print(f"1 AND 1 -> Fiber 2 Activation: {q_out[2].item():.4f}")
    if q_out[2] > 0.5:
        print("SUCCESS: 1 AND 1 -> 1")
    else:
        print("FAILURE: 1 AND 1 -> 0 (Did not learn)")
        
    # Check Weights
    J = executor.resonator.coupling
    print("\nCoupling Matrix J (Subset):")
    print(J.data[:3, :3])
    
    # Expectation: J[2,0] > 0, J[2,1] > 0.
    # Also likely J[0,2] > 0 (symmetric Hebbian).

if __name__ == "__main__":
    test_eq_prop_logic_gate()
