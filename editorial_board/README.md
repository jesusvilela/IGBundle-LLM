# Agentic Editorial Board + Publisher Router

This module implements a multi-agent system for automating the peer review, editing, and publishing process for scientific manuscripts, specifically targeting venues like arXiv.

## Structure

- **`agents.py`**: Defines the agent roles:
  - `EditorialAgent` (Reviewers): Conducts domain-specific reviews (Methodology, Reproducibility, etc.).
  - `MetaReviewer` (Area Chair): Consolidates reviews into a decision.
  - `ScientificEditor`: Polishes the manuscript text.
  - `ProductionAgent`: Handles LaTeX compilation and artifact generation.
  - `PublisherRouter`: Selects appropriate venues.
  - `PublisherConnector`: Simulates deposition to venues.
- **`state.py`**: Defines data contracts (`ManuscriptPackage`, `ReviewResult`, `PublicationArtifact`).
- **`tools.py`**: Provides utilities for normalization, citation checking (mock), and compilation.
- **`orchestrator.py`**: The main entry point to run the pipeline.

## Usage

To run the full pipeline:

```bash
python -m igbundle-llm.editorial_board.orchestrator
# OR if inside igbundle-llm:
python editorial_board/orchestrator.py
```

## Workflow

1.  **Intake**: Manuscript is normalized.
2.  **Review**: Parallel agents review the paper.
3.  **Decision**: Meta-reviewer accepts, rejects, or requests revision.
4.  **Edit**: Scientific Editor polishes the content.
5.  **Production**: PDF and Metadata are generated.
6.  **Publish**: Artifacts are deposited to selected venues (e.g., arXiv, Zenodo).

## Dependencies

- Standard Python libraries (asyncio, dataclasses).
- `pdflatex` (optional, for real PDF generation).
- `llmos` core tools (expected in parent directory).