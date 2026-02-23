# Poincaré Manifold Projection - User Guide

## What Am I Looking At?

The circular visualization in Neural Glass shows where the AI's "thoughts" are happening in a special kind of space called **hyperbolic geometry**. Think of it as a map of the AI's reasoning process.

---

## The Zones

### 🟢 **GREEN CENTER (Anchor Zone)**
- **What it means**: The AI is confident and stable
- **Behavior**: Fast, intuitive responses (like human "gut feelings")
- **Technical**: System 1 thinking, low entropy, familiar semantic territory

### 🟡 **YELLOW RING (Balanced Zone)**  
- **What it means**: The AI is weighing options
- **Behavior**: Considering multiple possibilities, moderate certainty
- **Technical**: Transitional state between intuition and analysis

### 🟠 **ORANGE RING (Exploratory Zone)**
- **What it means**: Deep analysis happening
- **Behavior**: Careful, deliberate reasoning (like solving a puzzle)
- **Technical**: System 2 engaged, higher uncertainty, creative exploration

### 🔵 **CYAN BOUNDARY (Stability Limit)**
- **What it means**: DANGER ZONE - semantic instability
- **Behavior**: If thoughts go here, meaning can "break"
- **Technical**: Hyperbolic infinity, exponential distance growth

---

## The Trajectory (Magenta Line)

The **magenta line with dots** shows how the AI's thinking evolved:

- **Dim/small dots** = Older thoughts
- **Bright/large dots** = Recent thoughts  
- **⭐ White star** = Current position

Watch how the trajectory moves:
- **Spiraling inward** → Converging on an answer
- **Spiraling outward** → Exploring possibilities
- **Jumping around** → Considering different approaches
- **Stuck in one spot** → Confident/repetitive

---

## The Numbers at the Bottom

### β (Beta) - Gibbs Temperature
This measures how "deterministic" vs "creative" the system is.

| Value | Meaning |
|-------|---------|
| β > 1.87 ✓ | "Quantum regime" - complex reasoning possible |
| β < 1.87 ✗ | Classical regime - simpler dynamics |

**Higher β** = More focused, deterministic  
**Lower β** = More exploratory, creative

Our default (β ≈ 4.6) is well into the "quantum advantage" regime.

### κ (Kappa) - Sectional Curvature
This measures the "shape" of the thought space.

| Value | Meaning |
|-------|---------|
| κ = -1.0 | Standard hyperbolic (normal) |
| κ < -1.0 | More curved (tighter reasoning) |
| κ → 0 | Flattening (simpler reasoning) |

---

## Why Hyperbolic Geometry?

In hyperbolic space, **distances grow exponentially** toward the boundary. This means:

1. **Near the center**: Small movements = small changes in meaning
2. **Near the edge**: Tiny movements = HUGE semantic shifts

This is why the AI can sometimes seem to "teleport" in meaning - a small step near the boundary creates massive change.

**The key insight**: This geometry naturally encodes the difference between:
- **Easy, familiar concepts** (center, close together)
- **Complex, nuanced concepts** (edge, far apart)

---

## What to Watch For

### 🟢 Healthy Patterns
- Trajectory stays mostly in green/yellow zones
- Smooth, continuous movement
- Gradual convergence when answering

### 🟡 Interesting Patterns  
- Brief excursions to orange zone during complex reasoning
- Spiral patterns when exploring options
- Return to center after exploration

### 🔴 Warning Signs
- Trajectory hitting the boundary repeatedly
- Chaotic jumping without convergence
- Stuck at the edge (semantic instability)

---

## In Plain English

**The Poincaré disk is like a "confidence meter" for AI reasoning:**

- **Center** = "I know this well"
- **Middle** = "Let me think about this"
- **Edge** = "This is really complex/uncertain"
- **Beyond edge** = "I'm confused" (system prevents this)

The trajectory shows you the AI's "train of thought" - where it started, where it explored, and where it ended up.

---

*Document version: February 2026 | ManifoldGL Phase 2*
