import logging
import asyncio
from typing import List
from .agents import (
    EditorialAgent, MetaReviewer,
    ScientificEditor, ProductionAgent, 
    PublisherRouter, PublisherConnector
)
from .state import ManuscriptPackage
from .tools import EditorialTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("workflow")

class EditorialBoard:
    def __init__(self):
        # We use EditorialAgent for generic roles where specific classes aren't defined
        self.intake = EditorialAgent("Intake-1", "Intake") 
        self.reviewers = [
            EditorialAgent("Rev-Method", "Methodology Reviewer"),
            EditorialAgent("Rev-Repro", "Reproducibility Reviewer"),
            EditorialAgent("Rev-Domain", "Domain Reviewer"),
            EditorialAgent("Rev-Cite", "Literature Reviewer"),
            EditorialAgent("Rev-Clear", "Clarity Reviewer"),
        ]
        self.meta_reviewer = MetaReviewer("Meta-1", "Area Chair")
        self.editor = ScientificEditor("Editor-1", "Scientific Editor")
        self.auditor = EditorialAgent("Audit-1", "Verification Agent")
        self.production = ProductionAgent("Prod-1", "Typesetter")
        self.router = PublisherRouter("Router-1", "Venue Router")
        # PublisherConnector is instantiated per venue usually, but we can keep a default
        self.publisher = PublisherConnector("arXiv") # Default

    async def process_manuscript(self, file_path: str):
        logger.info(f"=== Starting Editorial Process for {file_path} ===")
        
        # 1. Intake
        content = EditorialTools.normalize_manuscript(file_path)
        pkg = ManuscriptPackage(
            title="ManifoldGL: Information-Geometric Bundle Adapters",
            content=content,
            authors=["J. Vilela Jato", "A.I. Scientist", "H.A.L. 9000"],
            citations=["arXiv:2501.0001"] # Mock
        )
        logger.info(f"Intake complete: {pkg.title}")
        
        # 2. Review Loop
        for round_num in range(1, 3):
            logger.info(f"--- Review Round {round_num} ---")
            
            # Parallel Reviews
            review_tasks = [rev.review_manuscript(pkg) for rev in self.reviewers]
            reviews = await asyncio.gather(*review_tasks)
            
            # Meta Review
            decision = await self.meta_reviewer.consolidate(reviews)
            logger.info(f"Decision: {decision.decision.value}")
            logger.info(f"Synthesis: {decision.synthesis}")
            
            if "REJECT" in str(decision.decision).upper():
                logger.info("Manuscript Rejected.")
                return
            
            if "ACCEPT" in str(decision.decision).upper():
                break
                
            # If Revision needed
            logger.info("Performing Revisions...")
            pkg = await self.editor.polish(pkg)
            
            # Simple simulation: assume fixed after 1 round
            if round_num == 1:
                logger.info("Revisions complete. Resubmitting...")
        
        # 3. Audit (Simulated)
        logger.info("Auditing final package...")
        await asyncio.sleep(0.5)
        
        # 4. Production
        venues = await self.router.route(pkg)
        logger.info(f"Selected Venues: {venues}")
        
        for venue in venues:
             # 5. Connect & Deposit
            artifact = await self.production.produce_artifact(pkg, venue)
            connector = PublisherConnector(venue)
            await connector.deposit(artifact)
            logger.info(f"Published to {venue}: {artifact.status}")

        logger.info("=== Process Complete ===")

async def run_demo():
    board = EditorialBoard()
    # Ensure tool can handle non-existent dummy file if normalization fails?
    # Actually EditorialTools.normalize_manuscript likely just reads the file.
    await board.process_manuscript("submission_package/ManifoldGL_arXiv_submission.tex")

if __name__ == "__main__":
    asyncio.run(run_demo())
