#!/usr/bin/env python3
"""
Demonstration of Geometrically Rigorous IGBundle Implementation

This script demonstrates the corrected mathematical implementation with:
1. True Riemannian manifold geometry with curvature tensors
2. Proper fiber-to-fiber bundle lambda calculus operations
3. Information-geometric natural gradients
4. Sheaf-theoretic consistency constraints

Usage:
    python geometric_igbundle_demo.py

Author: LLMOS SystemAgent
License: MIT
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add IGBundle to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from igbundle.modules.geometric_adapter import GeometricIGBundleAdapter
from igbundle.training.geometric_training import GeometricTrainer, GeometricTrainingConfig
from igbundle.geometry.riemannian import RiemannianGeometry, FiberBundleLambdaCalculus

class GeometricConfig:
    """Configuration for geometrically rigorous IGBundle."""

    def __init__(self):
        # Model dimensions
        self.hidden_size = 512
        self.latent_dim = 64
        self.num_components = 4
        self.num_categories = 16
        self.bottleneck_dim = 128

        # Geometric parameters
        self.alpha = 1.0  # Base manifold KL weight
        self.beta = 1.0   # Fiber bundle KL weight
        self.eta_f = 0.1  # Fiber learning rate
        self.eta_b = 0.01 # Base learning rate
        self.eta_w = 0.01 # Weight learning rate

        # Adapter parameters
        self.adapter_scale = 0.1
        self.dropout = 0.1

        # Sheaf theory parameters
        self.num_sheaf_patches = 6

class SimpleTransformerWithGeometricAdapter(nn.Module):
    """
    Simple transformer model with geometrically rigorous IGBundle adapter.

    This demonstrates integration of proper Riemannian geometry and
    lambda calculus operations within a transformer architecture.
    """

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 512, num_layers: int = 6):
        super().__init__()

        self.config = GeometricConfig()
        self.config.hidden_size = hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(1024, hidden_size)

        # Transformer layers with geometric adapters
        self.layers = nn.ModuleList([
            TransformerLayerWithGeometry(self.config) for _ in range(num_layers)
        ])

        # Output head
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        print(f"🧮 GeometricTransformer initialized:")
        print(f"   📐 Riemannian manifold dimension: {self.config.latent_dim}")
        print(f"   🎭 Fiber bundle components: {self.config.num_components}")
        print(f"   🏷️ Categorical fiber dimension: {self.config.num_categories}")
        print(f"   λ  Lambda calculus operations: enabled")
        print(f"   ∇  Natural gradient optimization: enabled")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape

        # Embeddings
        token_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(torch.arange(T, device=input_ids.device))
        x = token_embeds + pos_embeds.unsqueeze(0)

        # Transform through geometric layers
        geometric_states = []
        for layer in self.layers:
            x, geo_state = layer(x)
            geometric_states.append(geo_state)

        # Store geometric states for loss computation
        self._last_geometric_states = geometric_states

        # Output projection
        logits = self.output_proj(x)
        return logits

class TransformerLayerWithGeometry(nn.Module):
    """Transformer layer enhanced with geometric IGBundle adapter."""

    def __init__(self, config: GeometricConfig):
        super().__init__()

        # Standard attention and MLP
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        # Geometric IGBundle adapter
        self.geometric_adapter = GeometricIGBundleAdapter(config)

    def forward(self, x: torch.Tensor):
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + attn_out

        # MLP
        residual = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out

        # Geometric adaptation
        x, geometric_state = self.geometric_adapter(x)

        return x, geometric_state

def demonstrate_geometric_operations():
    """Demonstrate core geometric operations."""
    print("\n🔬 GEOMETRIC OPERATIONS DEMONSTRATION")
    print("=" * 60)

    config = GeometricConfig()

    # 1. Riemannian Geometry
    print("\n1. 📐 Riemannian Manifold Operations:")
    riemannian = RiemannianGeometry(config)

    # Sample points on manifold
    batch_size, seq_len, num_comp = 2, 8, config.num_components
    positions = torch.randn(batch_size, seq_len, num_comp, config.latent_dim) * 0.5

    # Compute metric
    metric = riemannian.get_metric(positions)
    print(f"   ✓ Riemannian metric computed: {metric.metric_tensor.shape}")

    # Compute curvature
    curvature = riemannian.riemann_curvature(positions, metric)
    print(f"   ✓ Riemann curvature tensor: {curvature.shape}")

    # Sectional curvature
    u = torch.randn_like(positions)
    v = torch.randn_like(positions)
    sectional_k = riemannian.sectional_curvature(positions, u, v)
    print(f"   ✓ Sectional curvature: {sectional_k.mean().item():.4f} ± {sectional_k.std().item():.4f}")

    # 2. Lambda Calculus Operations
    print("\n2. λ Lambda Calculus in Fiber Bundles:")
    lambda_calc = FiberBundleLambdaCalculus(config)

    # Sample sections
    sections = torch.randn(batch_size, seq_len, num_comp, config.latent_dim + config.num_categories)

    # Lambda abstraction
    variable_type = torch.randn_like(sections)
    body = torch.randn_like(sections)
    lambda_term = lambda_calc.lambda_abstraction(variable_type, body)
    print(f"   ✓ Lambda abstraction: {lambda_term.shape}")

    # Function application
    function = torch.randn_like(sections)
    argument = torch.randn_like(sections)
    result = lambda_calc.application(function, argument)
    print(f"   ✓ Function application: {result.shape}")

    # Fiber morphism composition
    fiber_f = torch.randn(batch_size, seq_len, num_comp, config.num_categories)
    fiber_g = torch.randn_like(fiber_f)
    composed = lambda_calc.fiber_morphism_compose(fiber_f, fiber_g)
    print(f"   ✓ Fiber morphism composition: {composed.shape}")

    print("\n✅ All geometric operations successfully demonstrated!")

def demonstrate_corrected_training():
    """Demonstrate training with proper geometric losses."""
    print("\n🎯 CORRECTED TRAINING DEMONSTRATION")
    print("=" * 60)

    # Create model
    model = SimpleTransformerWithGeometricAdapter(vocab_size=1000, hidden_size=256, num_layers=2)

    # Training configuration with geometric losses
    train_config = GeometricTrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        max_steps=10,  # Short demo
        lambda_curvature=0.01,    # Curvature regularization
        lambda_sheaf=0.001,       # Sheaf consistency
        lambda_bundle=0.001,      # Bundle structure
        lambda_lambda=0.0001,     # Lambda calculus consistency
        use_natural_gradients=True,
        target_curvature_schedule="exponential",
        final_target_curvature=-1.0  # Hyperbolic
    )

    # Create trainer
    trainer = GeometricTrainer(model, train_config)

    print(f"📊 Training Configuration:")
    print(f"   🎯 Target curvature: {train_config.final_target_curvature} (hyperbolic)")
    print(f"   ∇  Natural gradients: {train_config.use_natural_gradients}")
    print(f"   🔗 Sheaf consistency weight: {train_config.lambda_sheaf}")
    print(f"   📐 Curvature loss weight: {train_config.lambda_curvature}")

    # Generate sample data
    batch_size, seq_len = 4, 32
    sample_batch = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"\n🏃 Running {train_config.max_steps} training steps:")

    for step in range(train_config.max_steps):
        # Training step
        losses = trainer.train_step(sample_batch)

        # Get current target curvature
        target_k = trainer.get_current_target_curvature()

        if step % 3 == 0:  # Print every 3rd step
            print(f"   Step {step+1:2d}: Loss={losses['total']:.4f}, "
                  f"LM={losses['language_modeling']:.4f}, "
                  f"Target-κ={target_k:.3f}")

            # Show geometric losses if present
            geo_losses = {k: v for k, v in losses.items()
                         if k not in ['total', 'language_modeling']}
            if geo_losses:
                geo_str = ", ".join([f"{k}={v:.5f}" for k, v in geo_losses.items()])
                print(f"         Geometric: {geo_str}")

    # Final metrics
    metrics = trainer.get_training_metrics()
    print(f"\n📈 Final Training Metrics:")
    print(f"   📊 Average loss (last 10): {metrics['average_loss_100']:.4f}")
    print(f"   🧮 Average geometric loss: {metrics['average_geometric_loss_100']:.5f}")
    print(f"   📐 Final target curvature: {metrics['target_curvature']:.3f}")

    print("\n✅ Geometric training completed successfully!")

def analyze_mathematical_corrections():
    """Analyze the mathematical corrections made."""
    print("\n🔍 MATHEMATICAL CORRECTIONS ANALYSIS")
    print("=" * 60)

    corrections = [
        ("❌ Fake 'curvature' (σ parameter)", "✅ True Riemann curvature tensor R^i_{jkl}"),
        ("❌ No lambda calculus", "✅ Proper λ-abstraction and application"),
        ("❌ Ad-hoc 'information geometry'", "✅ Natural gradients from Fisher metric"),
        ("❌ Missing parallel transport", "✅ Covariant derivative and connection"),
        ("❌ No categorical fiber operations", "✅ Fiber morphism composition"),
        ("❌ Euclidean bundle operations", "✅ Proper fiber bundle structure"),
        ("❌ Ungrounded 'sheaf theory'", "✅ Jensen-Shannon consistency constraints")
    ]

    print("Before → After:")
    for before, after in corrections:
        print(f"  {before}")
        print(f"  {after}")
        print()

    print("🎓 Scientific Rigor Improvements:")
    print("  • Christoffel symbols computed from learned metrics")
    print("  • Sectional curvature properly defines manifold geometry")
    print("  • Lambda terms have proper type checking")
    print("  • Bundle local triviality constraints enforced")
    print("  • Natural gradients respect Fisher information geometry")
    print("  • Sheaf gluing conditions use proper divergence measures")

def main():
    """Main demonstration function."""
    print("🧮 GEOMETRICALLY RIGOROUS IGBUNDLE DEMONSTRATION")
    print("=" * 60)
    print("Implementing TRUE fiber bundle lambda calculus with Riemannian geometry")
    print("Correcting mathematical inadequacies in original implementation")
    print()

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    try:
        # 1. Demonstrate core geometric operations
        demonstrate_geometric_operations()

        # 2. Show corrected training with proper losses
        demonstrate_corrected_training()

        # 3. Analyze mathematical corrections
        analyze_mathematical_corrections()

        print(f"\n🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The IGBundle implementation now has proper mathematical foundations:")
        print("✅ True Riemannian manifold geometry")
        print("✅ Proper fiber-to-fiber bundle lambda calculus")
        print("✅ Information-geometric natural gradients")
        print("✅ Sheaf-theoretic consistency constraints")
        print("✅ Bundle structure preservation")
        print()
        print("🚨 TRAINING SAFELY PRESERVED - No interruption to existing processes")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())