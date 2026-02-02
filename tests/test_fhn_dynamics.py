
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add path
sys.path.append(os.path.abspath("h:/LLM-MANIFOLD/igbundle-llm/src"))

from igbundle.dynamics.fhn import FHNIntegrator

def test_fhn_oscillation():
    print("--- Testing FHN Oscillation ---")
    num_fibers = 2
    fhn = FHNIntegrator(num_fibers=num_fibers, dt=0.05)
    
    # Initial State: Rest
    q0 = torch.zeros(1, num_fibers)
    p0 = torch.zeros(1, num_fibers)
    
    # Input Drive: Fiber 0 gets a pulse, Fiber 1 gets nothing
    steps = 500
    drive = torch.zeros(1, steps, num_fibers)
    drive[:, 50:150, 0] = 1.0  # Pulse for 100 steps
    
    # Simulate
    q_trace, (q_final, p_final) = fhn.simulate((q0, p0), drive)
    
    print(f"Trajectory Shape: {q_trace.shape}")
    
    # Check if Fiber 0 excited (Max q > 0.5)
    max_q0 = q_trace[0, :, 0].max().item()
    print(f"Fiber 0 Max Excitation: {max_q0:.4f}")
    if max_q0 > 0.5:
        print("SUCCESS: Fiber 0 excited by input.")
    else:
        print("FAILURE: Fiber 0 did not excite.")
        
    # Check Fiber 1 (Coupling check)
    # With random init coupling * 0.01, specific excitation is unlikely to be strong,
    # but we verify it runs.
    max_q1 = q_trace[0, :, 1].max().item()
    print(f"Fiber 1 Max Excitation (Coupled): {max_q1:.4f}")
    
    return q_trace

if __name__ == "__main__":
    test_fhn_oscillation()
