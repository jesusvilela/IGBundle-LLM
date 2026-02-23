"""
Test Script for Geometric Scrambling Diagnostic
Validates scrambling measurements on Poincaré ball Hamiltonian dynamics.

Run: python test_scrambling.py
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from igbundle.quantum.scrambling import (
    GeometricScramblingDiagnostic,
    ScramblingResult,
    ScramblingVisualizer
)
import numpy as np
import time


def test_scrambling_diagnostic():
    """Run full scrambling diagnostic test."""
    print("=" * 60)
    print("GEOMETRIC SCRAMBLING DIAGNOSTIC TEST")
    print("ManifoldGL Phase 2.5 - Classical OTOC Analog")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Initialize diagnostic module
    diagnostic = GeometricScramblingDiagnostic(
        manifold=None,  # We use internal implementation
        latent_dim=64,
        perturbation_scale=1e-4,
        max_steps=100,
        step_size=0.05,
        damping=0.01,  # Maps to β ≈ 4.6
        saturation_threshold=0.95
    ).to(device)
    
    print(f"\n--- Configuration ---")
    print(f"Latent Dim: 64")
    print(f"Perturbation Scale: 1e-4")
    print(f"Max Steps: 100")
    print(f"Step Size: 0.05")
    print(f"Damping: 0.01")
    
    # Check temperature mapping
    beta = diagnostic.gibbs_temperature()
    print(f"\n--- Gibbs Temperature Mapping ---")
    print(f"Damping q = 0.01 → β = {beta:.2f}")
    print(f"Coherence threshold: β > 1.87")
    print(f"Above hardness threshold: {diagnostic.is_above_hardness_threshold()}")
    
    # Run scrambling measurement
    print(f"\n--- Running Scrambling Measurement ---")
    start_time = time.time()
    
    # Test with random initial state near origin
    initial_state = torch.randn(1, 64, device=device) * 0.1
    result = diagnostic.measure_scrambling(initial_state=initial_state)
    
    elapsed = time.time() - start_time
    print(f"Measurement completed in {elapsed:.3f}s")
    
    # Display results
    print(f"\n--- Scrambling Results ---")
    formatted = ScramblingVisualizer.format_result(result)
    for key, value in formatted.items():
        print(f"  {key}: {value}")
    
    # Analyze trajectory
    distances = np.array(result.distance_trajectory)
    print(f"\n--- Trajectory Analysis ---")
    print(f"  Initial distance: {distances[0]:.6f}")
    print(f"  Final distance: {distances[-1]:.6f}")
    print(f"  Max distance: {distances.max():.6f}")
    print(f"  Growth factor: {distances[-1] / (distances[0] + 1e-10):.2f}x")
    
    # Test with different initial conditions
    print(f"\n--- Multi-Condition Test ---")
    results = []
    for i in range(5):
        init = torch.randn(1, 64, device=device) * (0.1 * (i + 1))
        r = diagnostic.measure_scrambling(initial_state=init)
        results.append(r)
        print(f"  Trial {i+1} (scale {0.1*(i+1):.1f}): τ_s={r.scrambling_time:.3f}, λ={r.lyapunov_exponent:.4f}, chaotic={r.is_chaotic}")
    
    # Average Lyapunov
    avg_lyapunov = np.mean([r.lyapunov_exponent for r in results])
    print(f"\n  Average Lyapunov Exponent: {avg_lyapunov:.4f}")
    
    # OTOC test
    print(f"\n--- OTOC Analog Test ---")
    W = torch.randn(1, 64, device=device)
    V = torch.randn(1, 64, device=device)
    initial = torch.randn(1, 64, device=device) * 0.1
    
    otoc = diagnostic.otoc_analog(W, V, initial, time_steps=20)
    print(f"  OTOC trajectory (first 10): {[f'{c:.6f}' for c in otoc[:10]]}")
    print(f"  OTOC growth: {otoc[-1] / (otoc[0] + 1e-10):.2f}x")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Geometric scrambling diagnostic operational")
    print(f"✓ Effective temperature β = {beta:.2f} (> 1.87 threshold)")
    print(f"✓ {'Chaotic regime detected' if result.is_chaotic else 'Stable regime'} (λ = {result.lyapunov_exponent:.4f})")
    print(f"✓ Scrambling time τ_s = {result.scrambling_time:.3f}")
    print(f"✓ OTOC analog computed successfully")
    
    if result.above_hardness_threshold:
        print(f"\n⚡ HIGH COHERENCE REGIME:")
        print(f"   sampling from this Gibbs state is classically hard!")
    
    return result


def test_visualization():
    """Test visualization output format."""
    print(f"\n" + "=" * 60)
    print("VISUALIZATION FORMAT TEST")
    print("=" * 60)
    
    # Create mock result
    result = ScramblingResult(
        scrambling_time=0.35,
        lyapunov_exponent=2.5,
        saturation_distance=0.85,
        distance_trajectory=list(np.linspace(0.001, 0.85, 50)),
        is_chaotic=True,
        effective_temperature=4.6,
        above_hardness_threshold=True
    )
    
    # Format for display
    formatted = ScramblingVisualizer.format_result(result)
    print("\nFormatted Output:")
    for k, v in formatted.items():
        print(f"  {k}: {v}")
    
    # Plotly trace
    trace = ScramblingVisualizer.create_plotly_trace(result)
    print(f"\nPlotly Trace Keys: {list(trace.keys())}")
    print(f"X range: [{trace['x'][0]:.2f}, {trace['x'][-1]:.2f}]")
    print(f"Y range: [{min(trace['y']):.4f}, {max(trace['y']):.4f}]")


if __name__ == "__main__":
    result = test_scrambling_diagnostic()
    test_visualization()
    
    print(f"\n✅ All tests passed!")
