
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports if run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from editorial_board.state import ManuscriptPackage, ReviewDecision
from editorial_board.agents import (
    EditorialAgent, MetaReviewer, ScientificEditor, 
    ProductionAgent, PublisherRouter, PublisherConnector
)
from editorial_board.tools import EditorialTools

async def run_editorial_pipeline(manuscript_path: str):
    print("\n=== STARTING AGENTIC EDITORIAL BOARD PIPELINE ===\n")

    # 1. Intake
    content = EditorialTools.normalize_manuscript(manuscript_path)
    manuscript = ManuscriptPackage(
        title="Automated Discovery of Manifold Curvature in LLM Latent Spaces",
        content=content,
        authors=["A. I. Scientist", "H. A. L. 9000"],
        citations=["arXiv:2501.0001", "doi:10.1038/nature12345"]
    )
    print(f"Loaded manuscript: {manuscript.title}")

    # 2. Review Board Assembly
    reviewers = [
        EditorialAgent("Gauss", "Methodology Reviewer"),
        EditorialAgent("Popper", "Reproducibility Reviewer"),
        EditorialAgent("Riemann", "Domain Reviewer (Geometry)"),
        EditorialAgent("Borges", "Literature Reviewer"),
    ]
    meta_reviewer = MetaReviewer("Hilbert", "Area Chair")
    
    # 3. Parallel Review Process
    print("\n--- STAGE 1: PARALLEL PEER REVIEW ---")
    review_tasks = [r.review_manuscript(manuscript) for r in reviewers]
    reviews = await asyncio.gather(*review_tasks)
    
    for rev in reviews:
        print(f"\n[Review] {rev.reviewer_name} ({rev.role}):")
        print(f"  Score: {rev.scores}")
        print(f"  Summary: {rev.summary}")
        if rev.issues:
            print(f"  Issues: {len(rev.issues)} found.")

    # 4. Meta-Review
    print("\n--- STAGE 2: META-REVIEW & DECISION ---")
    decision_package = await meta_reviewer.consolidate(reviews)
    print(f"Decision: {decision_package.decision.value}")
    print(f"Synthesis: {decision_package.synthesis}")
    
    if decision_package.decision == ReviewDecision.REJECT:
        print("Pipeline terminated: Manuscript rejected.")
        return

    if decision_package.decision in [ReviewDecision.MAJOR_REVISION, ReviewDecision.MINOR_REVISION]:
        print("Pipeline paused: Revision required.")
        # In a full loop, we would trigger the author agent here. 
        # For this demo, we simulate a fast-forward fix
        print(">> SIMULATING REVISION CYCLE >>")
        manuscript.version += 1
    
    # 5. Scientific Editing
    print("\n--- STAGE 3: SCIENTIFIC EDITING ---")
    editor = ScientificEditor("Hemingway", "Scientific Editor")
    manuscript = await editor.polish(manuscript)
    print(f"Polished Title: {manuscript.title}")

    # 6. Production & Routing
    print("\n--- STAGE 4: PRODUCTION & PUBLISHING ---")
    prod_agent = ProductionAgent("Knuth", "Typesetter")
    router = PublisherRouter("Maxwell", "Venue Router")
    
    venues = await router.route(manuscript)
    print(f"Selected Venues: {venues}")
    
    for venue in venues:
        # Create Artifact
        artifact = await prod_agent.produce_artifact(manuscript, venue)
        
        # Connect & Deposit
        connector = PublisherConnector(venue)
        await connector.deposit(artifact)
        print(f"Status for {venue}: {artifact.status}")

    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    # Ensure dummy file exists
    dummy_path = "igbundle-llm/dummy_manuscript.tex"
    if not os.path.exists(dummy_path):
        with open(dummy_path, "w") as f:
            f.write("Dummy LaTeX Content")
            
    asyncio.run(run_editorial_pipeline(dummy_path))
