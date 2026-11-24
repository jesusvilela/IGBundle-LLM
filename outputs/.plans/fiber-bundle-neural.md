# Research Plan: Fiber Bundle Geometry in Neural Networks — recent advances (2023-2026), empirical results, and connections to fiber bundle geometry in transformers

## Questions
1. What recent (2023-2026) papers explicitly use fiber bundle geometry, principal bundles, associated bundles, gauge structure, or bundle-like constructions in neural networks?
2. Which of those works are theoretical only, and which report empirical results on modern neural architectures or benchmarks?
3. What empirical outcomes are actually reported: accuracy, robustness, sample efficiency, representation quality, optimization behavior, equivariance, interpretability, or scaling effects?
4. Which claims are supported by public code, reproducible experiments, or benchmark tables rather than conceptual framing alone?
5. How are bundle concepts mapped into neural-network objects (layers, activations, features, latent variables, parameter spaces, token representations, attention heads, symmetries, or gauge fields)?
6. What work connects fiber bundle geometry specifically to transformers, attention, sequence models, or token manifolds?
7. Are there adjacent but relevant lines of work—gauge-equivariant networks, geometric deep learning, connection-based representation learning, manifold learning, information geometry—that should be treated as supporting context rather than direct fiber-bundle evidence?
8. What disagreements, gaps, or failure modes appear across the literature: weak empirical grounding, overloaded terminology, missing ablations, or lack of transformer-specific demonstrations?
9. What evidence exists in public repositories, project pages, or docs that bundle-inspired transformer claims have been implemented beyond speculative proposals?
10. What conclusions are justified today about the maturity of fiber-bundle geometry in neural networks, especially for transformers?

## Strategy
- Scope: prioritize 2023-2026 primary sources, but allow pre-2023 foundational papers only when needed to interpret recent claims.
- Evidence priority:
  1. Peer-reviewed papers and arXiv preprints with experiments
  2. Official project/repository pages and code
  3. Benchmark tables, appendices, and implementation docs
  4. Recent workshop talks/blogs only if they point to primary artifacts
- Researcher allocations and dimensions:
  - R1 / T1-T2: academic paper sweep for direct fiber-bundle neural-network papers (2023-2026), focusing on definitions, architectures, and experiments.
  - R2 / T3-T4: transformer-specific sweep, including attention/token geometry, bundle formulations, and whether transformer claims are empirical or speculative.
  - R3 / T5-T6: code/repository audit for publicly available implementations, benchmark evidence, datasets, and reproducibility signals.
  - R4 / T7-T8: adjacent literature boundary-setting—gauge-equivariant and geometric-learning work that is often conflated with fiber bundles; identify what is truly bundle-specific versus merely geometric.
- Expected rounds:
  - Round 1: broad evidence gathering across four disjoint angles.
  - Round 2: targeted gap-filling on contradictions, thin empirical support, or missing transformer evidence.
- Synthesis method:
  - Build a source matrix with: claim type, bundle formalism, architecture class, empirical setting, code availability, and transformer relevance.
  - Separate observations from inferences.
  - Downgrade unsupported claims, especially where papers use bundle language metaphorically.
- Output target:
  - One cited research brief with an evidence-backed maturity assessment and explicit treatment of transformers.

## Acceptance Criteria
- [ ] All key questions answered with ≥2 independent sources where feasible
- [ ] Direct fiber-bundle papers from 2023-2026 identified and categorized
- [ ] Transformer-specific connections assessed using primary sources rather than analogy alone
- [ ] Empirical claims summarized with actual tasks/metrics, not only narrative descriptions
- [ ] Code availability and reproducibility status checked for major empirical papers
- [ ] Contradictions, terminology slippage, and evidence gaps identified and addressed
- [ ] No single-source claims on critical findings unless explicitly labeled as tentative
- [ ] Final brief distinguishes direct evidence, adjacent evidence, and inference

## Task Ledger
| ID | Owner | Task | Status | Output |
|---|---|---|---|---|
| T1 | lead / researcher | Find 2023-2026 papers explicitly framing neural networks with fiber bundles or closely equivalent bundle formalisms | todo | outputs/fiber-bundle-neural-research-papers.md |
| T2 | lead / researcher | Extract architecture type, mathematical construction, and empirical setting from direct bundle papers | todo | outputs/fiber-bundle-neural-research-papers.md |
| T3 | lead / researcher | Find transformer-specific papers, preprints, repos, or docs connecting fiber bundles to attention/token geometry | todo | outputs/fiber-bundle-neural-research-transformers.md |
| T4 | lead / researcher | Judge whether transformer connections are implemented, benchmarked, or still conceptual | todo | outputs/fiber-bundle-neural-research-transformers.md |
| T5 | lead / researcher | Audit public code/repos/project pages for direct empirical support and reproducibility signals | todo | outputs/fiber-bundle-neural-research-code.md |
| T6 | lead / researcher | Record benchmark tasks, metrics, ablations, and missing artifacts for main empirical works | todo | outputs/fiber-bundle-neural-research-code.md |
| T7 | lead / researcher | Map adjacent geometric/gauge literature that may inform or confound bundle claims | todo | outputs/fiber-bundle-neural-research-adjacent.md |
| T8 | lead / researcher | Separate direct fiber-bundle evidence from broader geometric deep learning context | todo | outputs/fiber-bundle-neural-research-adjacent.md |
| T9 | lead | Synthesize evidence, resolve contradictions, and draft brief | todo | outputs/.drafts/fiber-bundle-neural-draft.md |
| T10 | verifier | Add inline citations, verify URLs, and build sources list | todo | outputs/fiber-bundle-neural-brief.md |
| T11 | reviewer | Verification pass for unsupported claims, logic gaps, and confidence calibration | todo | outputs/fiber-bundle-neural-verification.md |
| T12 | lead | Finalize deliverable and provenance record | todo | outputs/fiber-bundle-neural.md |

## Verification Log
| Item | Method | Status | Evidence |
|---|---|---|---|
| Coverage of direct 2023-2026 fiber-bundle neural-network literature | source cross-read across paper sweep + citation chasing | pending | outputs/fiber-bundle-neural-research-papers.md |
| Transformer-specific bundle claims | direct fetch of papers/repos/docs + cross-read | pending | outputs/fiber-bundle-neural-research-transformers.md |
| Empirical benchmark numbers in final brief | table-to-source mapping and citation sweep | pending | outputs/.drafts/fiber-bundle-neural-draft.md |
| Code availability and reproducibility statements | repository inspection / official links | pending | outputs/fiber-bundle-neural-research-code.md |
| Boundary between direct bundle evidence and adjacent gauge/geometric work | source comparison across papers | pending | outputs/fiber-bundle-neural-research-adjacent.md |
| Final citations and URL validity | verifier pass | pending | outputs/fiber-bundle-neural-brief.md |
| Confidence calibration and unsupported-claim check | reviewer pass | pending | outputs/fiber-bundle-neural-verification.md |

## Decision Log
- Slug chosen: `fiber-bundle-neural`.
- Time emphasis fixed to 2023-2026, with older foundational sources allowed only for context.
- Planned scale: 4 parallel researcher subagents because the topic is a broad, multi-faceted survey with distinct evidence types.
- No `CHANGELOG.md` existed at planning time, so there were no prior lab-notebook entries to incorporate.
