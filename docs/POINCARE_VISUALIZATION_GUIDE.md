# Poincaré Manifold Projection - Visual Guide

## What You're Looking At

The circular visualization in Neural Glass is a **Poincaré disk projection** — a way to visualize infinite hyperbolic space within a finite circle. It shows where the model's "thoughts" are positioned in semantic space as it generates text.

```
                    ABSTRACT / THEORETICAL
                           ⬆
                           |
      EXPLORATORY   ╭──────────────╮   EXPLORATORY
        (System 2)  │              │   (System 2)
                    │   ╭──────╮   │
    ANALYTICAL ◄────│   │ CORE │   │────► CREATIVE
                    │   │      │   │
                    │   ╰──────╯   │
        BALANCED    │              │   BALANCED
                    ╰──────────────╯
                           |
                           ⬇
                    CONCRETE / PRACTICAL
```

## The Three Cognitive Zones

### 🟢 ANCHOR ZONE (Center, r < 0.3)
**Color:** Green glow  
**Meaning:** High confidence, stable semantics  
**Cognitive Style:** System 1 (fast, intuitive)

When the thought trace stays near the center:
- Model is confident in its response
- Drawing on well-established knowledge
- Low uncertainty, direct answers
- Examples: Factual recall, simple instructions

### 🟡 BALANCED ZONE (Middle ring, 0.3 < r < 0.6)
**Color:** Yellow glow  
**Meaning:** Weighing options, moderate certainty  
**Cognitive Style:** Transitional

When the thought trace orbits this zone:
- Model is considering multiple perspectives
- Evaluating trade-offs
- Balanced reasoning
- Examples: Comparisons, nuanced explanations

### 🟠 EXPLORATORY ZONE (Outer ring, 0.6 < r < 0.85)
**Color:** Orange glow  
**Meaning:** Deep analysis, high uncertainty  
**Cognitive Style:** System 2 (slow, deliberate)

When the thought trace ventures here:
- Model is exploring novel territory
- High cognitive load
- Creative or speculative thinking
- Examples: Complex problem-solving, creative writing

### ⚠️ BOUNDARY ZONE (Near edge, r > 0.85)
**Color:** Cyan boundary line  
**Meaning:** Semantic instability risk  
**Warning:** Approaching coherence limits

When approaching the boundary:
- Model may be hallucinating
- Semantic coherence degrading
- Risk of nonsensical output
- The "Thesis Reset" may trigger if crossed

## Reading the Trajectory

### The Magenta Line
This shows the **path of thought** — how the model's semantic position evolved during generation.

| Pattern | Interpretation |
|---------|----------------|
| Tight spiral near center | Confident, focused response |
| Wide loops | Exploring alternatives |
| Sudden jumps | Topic shifts or "aha moments" |
| Drift toward edge | Increasing uncertainty |
| Return to center | Re-anchoring to core meaning |

### The White Star ⭐
This marks the **current position** — where the model is "thinking" right now.

### Point Size & Opacity
- **Larger, brighter points** = More recent thoughts
- **Smaller, dimmer points** = Earlier in the generation

## The Metrics Footer

At the bottom of the plot you'll see:
```
β=4.6 ✓ | κ=-1.0
```

### β (Gibbs Temperature)
Notes:
- **β > 1.87** (✓): System operates in "quantum advantage" regime
- **β < 1.87** (✗): Classical sampling regime

Higher β = more structured, coherent sampling.  
Our default β≈4.6 is well above the threshold.

### κ (Sectional Curvature)
- **κ = -1.0**: Pure hyperbolic (default Poincaré ball)
- **κ → 0**: Flattening toward Euclidean
- **κ > 0**: Would indicate spherical (not used)

Curvature affects how "distances" work:
- High negative curvature = thoughts diverge exponentially
- This enables rich semantic hierarchies

## What the Zones Tell You About Output Quality

| Zone | Output Characteristics | Example Queries |
|------|----------------------|-----------------|
| 🟢 Anchor | Direct, confident, factual | "What is 2+2?" |
| 🟡 Balanced | Nuanced, well-reasoned | "Compare Python vs Rust" |
| 🟠 Exploratory | Creative, speculative | "Write a poem about AI" |
| ⚠️ Boundary | Potentially unreliable | Adversarial prompts |

## The Bundle Indicator

The "ACTIVE BUNDLE" metric (e.g., "Bundle-5") shows which **fiber bundle** is currently active.

Think of bundles as specialized "reasoning modules":
- Different bundles activate for different cognitive tasks
- Rapid bundle switching = complex reasoning
- Stable bundle = focused processing

During our validation test, we observed:
```
Bundle-1 → Bundle-3 → Bundle-5 → Bundle-8 → Bundle-2
```
This 5-bundle sequence for a single query demonstrates the model engaging multiple cognitive strategies.

## Practical Interpretation

### Healthy Generation Pattern
```
╭────────╮
│  ⭐    │  Start near center (grounding)
│ ↙      │  Explore outward (analysis)
│↙       │  Return toward center (synthesis)
╰────────╯
```

### Warning Pattern
```
╭────────╮
│        │  
│       ⭐│  Stuck at edge = potential hallucination
│        │  
╰────────╯
```

### Creative Pattern
```
╭────────╮
│ ↗↘↗↘   │  
│  ⭐    │  Wide oscillations = exploring possibilities
│        │  
╰────────╯
```

## FAQ

**Q: Why is the space "hyperbolic"?**  
A: Hyperbolic geometry naturally represents hierarchical relationships. Semantic concepts form trees (hypernyms/hyponyms), and hyperbolic space can embed these with minimal distortion.

**Q: What causes jumps in the trajectory?**  
A: Topic changes, constraint violations (triggering neurosymbolic jumps), or the model switching between reasoning strategies.

**Q: Should I worry if it goes to the edge?**  
A: Brief excursions are normal during creative tasks. Sustained edge-dwelling, especially with decreasing Constraint Score, may indicate issues.

**Q: What's the difference between Entropy (S) and position?**  
A: Entropy measures uncertainty in the probability distribution over fibers. Position shows WHERE in semantic space. High entropy + edge position = maximum uncertainty.

---

*This guide accompanies the Neural Glass interface for ManifoldGL Phase 2.*
*Last updated: February 2026*
