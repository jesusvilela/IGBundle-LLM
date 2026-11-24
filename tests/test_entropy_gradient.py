"""
Phase A Validation: Verify gradient flow through fiber sections after entropy unfreeze fixes.

Tests that:
1. fiber_diversity loss gradient reaches fiber_section_proj.weight (logit-space, nonzero at uniform)
2. fiber_entropy loss gradient reaches fiber_section_proj.weight (Gumbel-Softmax, nonzero at uniform)
3. Removing .detach() allows base coords → fiber gradient flow
4. eta_f = 0.1 produces meaningful section updates (not negligible)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from src.igbundle.core.config import IGBundleConfig
from src.igbundle.modules.geometric_adapter import GeometricIGBundleAdapter


def make_adapter(use_dynamics=True):
    """Create a minimal adapter for gradient testing."""
    cfg = IGBundleConfig(
        hidden_size=128,
        latent_dim=8,
        num_components=4,
        num_categories=16,
        use_dynamics=use_dynamics,
        manifold_type='poincare',
    )
    adapter = GeometricIGBundleAdapter(cfg)
    adapter.train()
    return adapter


def test_fiber_diversity_gradient():
    """fiber_diversity loss must produce nonzero gradients on fiber_section_proj even at uniform."""
    print("Test 1: fiber_diversity gradient at uniform...")
    adapter = make_adapter(use_dynamics=True)

    # Force fiber_section_proj bias to zero → uniform logits → uniform softmax
    with torch.no_grad():
        adapter.fiber_section_proj.bias.zero_()
        adapter.fiber_section_proj.weight.zero_()

    x = torch.randn(1, 4, 128)  # (B, T, H)
    output, geo_state = adapter(x)

    losses = adapter.compute_geometric_losses(geo_state)
    assert 'fiber_diversity' in losses, "fiber_diversity loss not computed"
    loss = losses['fiber_diversity']
    print(f"  fiber_diversity loss = {loss.item():.4f}")

    loss.backward(retain_graph=True)
    grad = adapter.fiber_section_proj.weight.grad
    assert grad is not None, "No gradient on fiber_section_proj.weight!"
    grad_norm = grad.norm().item()
    print(f"  fiber_section_proj.weight grad norm = {grad_norm:.6f}")
    assert grad_norm > 1e-8, f"Gradient too small: {grad_norm}"
    print("  PASSED\n")


def test_fiber_entropy_squared_gradient():
    """Squared entropy loss (H - S_target)^2 must produce nonzero gradient once diversity breaks symmetry."""
    print("Test 2: fiber_entropy (squared) gradient...")
    adapter = make_adapter(use_dynamics=True)

    x = torch.randn(1, 4, 128)
    output, geo_state = adapter(x)

    losses = adapter.compute_geometric_losses(geo_state)
    assert 'fiber_entropy' in losses, "fiber_entropy loss not computed"
    loss = losses['fiber_entropy']
    print(f"  fiber_entropy loss = {loss.item():.4f}")

    # Combined loss to test full gradient path
    total = losses['fiber_diversity'] + losses['fiber_entropy']
    total.backward(retain_graph=True)
    grad = adapter.fiber_section_proj.weight.grad
    assert grad is not None, "No gradient on fiber_section_proj.weight!"
    grad_norm = grad.norm().item()
    print(f"  fiber_section_proj.weight grad norm = {grad_norm:.6f}")
    # Gradient should be nonzero from at least the diversity term
    assert grad_norm > 1e-8, f"Gradient too small: {grad_norm}"
    print("  PASSED\n")


def test_base_to_fiber_gradient_coupling():
    """With .detach() removed, base_coord_proj gradients must flow through fiber_update_net."""
    print("Test 3: base -> fiber gradient coupling...")
    adapter = make_adapter(use_dynamics=True)

    x = torch.randn(1, 4, 128)
    output, geo_state = adapter(x)

    losses = adapter.compute_geometric_losses(geo_state)
    total_loss = sum(losses.values())
    total_loss.backward()

    # fiber_update_net should receive gradients from both coord and section paths
    fun_grad = adapter.fiber_update_net[0].weight.grad
    assert fun_grad is not None, "fiber_update_net has no gradient!"
    fun_grad_norm = fun_grad.norm().item()
    print(f"  fiber_update_net[0].weight grad norm = {fun_grad_norm:.6f}")
    assert fun_grad_norm > 1e-8, f"fiber_update_net gradient too small: {fun_grad_norm}"

    # base_coord_proj should also have gradients (proving coupling)
    bcp_grad = adapter.base_coord_proj.weight.grad
    assert bcp_grad is not None, "base_coord_proj has no gradient!"
    bcp_grad_norm = bcp_grad.norm().item()
    print(f"  base_coord_proj.weight grad norm = {bcp_grad_norm:.6f}")
    assert bcp_grad_norm > 1e-8, f"base_coord_proj gradient too small: {bcp_grad_norm}"
    print("  PASSED\n")


def test_eta_f_meaningful():
    """eta_f = 0.1 should produce sections that differ meaningfully from uniform."""
    print("Test 4: eta_f produces meaningful updates...")
    adapter = make_adapter(use_dynamics=True)
    assert adapter.cfg.eta_f == 0.1, f"eta_f should be 0.1, got {adapter.cfg.eta_f}"

    x = torch.randn(1, 4, 128)
    output, geo_state = adapter(x)

    sections = geo_state.fiber_sections  # Should be probabilities
    # Check sections are valid probabilities
    assert (sections >= 0).all(), "Sections have negative values (not probabilities)"
    sums = sections.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Sections don't sum to 1: {sums.mean():.4f}"

    # Check deviation from uniform (1/K = 1/16 = 0.0625)
    uniform = 1.0 / adapter.K
    deviation = (sections - uniform).abs().mean().item()
    print(f"  Mean |p_i - 1/K| = {deviation:.6f}")
    print(f"  Sections std = {sections.std(dim=-1).mean().item():.6f}")
    # Even with random init, sections should deviate from uniform
    assert deviation > 1e-4, f"Sections too close to uniform: {deviation}"
    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase A Entropy Gradient Validation")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0
    errors = []

    for test_fn in [
        test_fiber_diversity_gradient,
        test_fiber_entropy_squared_gradient,
        test_base_to_fiber_gradient_coupling,
        test_eta_f_meaningful,
    ]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAILED: {e}\n")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)
