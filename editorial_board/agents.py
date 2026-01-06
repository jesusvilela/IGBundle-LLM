import asyncio
import logging
import os
from typing import List, Dict, Any, Union
from .state import ManuscriptPackage, ReviewResult, ReviewIssue, ConsolidatedReview, ReviewDecision, PublicationArtifact
from .tools import EditorialTools

# Configure Logging
logger = logging.getLogger("EditorialBoard")
logger.setLevel(logging.INFO)

class EditorialAgent:
    def __init__(self, name: str, role: str, model_mock_response: str = "default"):
        self.name = name
        self.role = role
        self.model_mock_response = model_mock_response

    async def review_manuscript(self, manuscript: ManuscriptPackage) -> ReviewResult:
        logger.info(f"[{self.role}] {self.name} is reviewing '{manuscript.title}'...")
        await asyncio.sleep(0.5) # Simulate processing time

        # Simulate analysis based on role
        scores = {}
        issues = []
        summary = ""

        if "Method" in self.role:
            scores = {"methodology": 8.0, "robustness": 7.5}
            summary = "Methods appear sound but lack detailed error analysis."
            issues.append(ReviewIssue("minor", "Missing error bars in Figure 2", "Section 3.1"))
        elif "Reproducibility" in self.role:
            scores = {"reproducibility": 6.0}
            summary = "Code availability is unclear."
            issues.append(ReviewIssue("major", "No link to code repository found.", "Abstract"))
        elif "Domain" in self.role:
            scores = {"novelty": 9.0, "significance": 8.5}
            summary = "Highly novel contribution to the field."
        elif "Literature" in self.role:
            scores = {"citation_coverage": 7.0}
            summary = "Misses recent work by Smith et al. (2025)."
            issues.append(ReviewIssue("minor", "Add citation to Smith 2025", "Introduction"))
        else:
            scores = {"general": 8.0}
            summary = "Good paper."

        return ReviewResult(
            reviewer_name=self.name,
            role=self.role,
            scores=scores,
            summary=summary,
            issues=issues,
            confidence=0.9
        )

class MetaReviewer(EditorialAgent):
    async def consolidate(self, reviews: List[ReviewResult]) -> ConsolidatedReview:
        logger.info(f"[{self.role}] {self.name} is consolidating {len(reviews)} reviews...")
        
        # Simple logic: if any major issues, Major Revision. Else Accept.
        major_issues = [i for r in reviews for i in r.issues if i.severity == "major"]
        
        if major_issues:
            decision = ReviewDecision.MAJOR_REVISION
            synthesis = "Significant issues found regarding reproducibility and citations."
            actions = [i.description for i in major_issues]
        else:
            decision = ReviewDecision.ACCEPT
            synthesis = "Strong paper, ready for publication with minor tweaks."
            actions = ["Address minor formatting issues."]

        return ConsolidatedReview(
            decision=decision,
            synthesis=synthesis,
            required_actions=actions,
            reviews=reviews
        )

class ScientificEditor(EditorialAgent):
    async def polish(self, manuscript: ManuscriptPackage) -> ManuscriptPackage:
        logger.info(f"[{self.role}] {self.name} is polishing the manuscript...")
        # Simulate text improvement
        manuscript.title = f"{manuscript.title} (Edited)"
        manuscript.metadata["editor_checked"] = "true"
        return manuscript

class ProductionAgent(EditorialAgent):
    async def produce_artifact(self, manuscript: ManuscriptPackage, target_venue: str) -> PublicationArtifact:
        logger.info(f"[{self.role}] {self.name} is generating publication artifacts for {target_venue}...")
        
        # Use tools
        output_dir = "hyperprotocol_pipeline_output/artifacts"
        pdf_path = EditorialTools.compile_latex("dummy_source.tex", output_dir)
        
        meta_path = os.path.join(output_dir, "metadata.json")
        EditorialTools.generate_metadata(manuscript.__dict__, meta_path)
        
        return PublicationArtifact(
            pdf_path=pdf_path,
            source_archive_path=output_dir,
            metadata_json_path=meta_path,
            target_venue=target_venue,
            status="ready"
        )

class PublisherRouter(EditorialAgent):
    async def route(self, manuscript: ManuscriptPackage) -> List[str]:
        logger.info(f"[{self.role}] {self.name} is selecting venues...")
        # Logic based on field/constraints
        return ["arXiv", "Zenodo"]

class PublisherConnector(EditorialAgent):
    def __init__(self, venue: str):
        super().__init__(f"{venue}Connector", "Publisher")
        self.venue = venue

    async def deposit(self, artifact: PublicationArtifact):
        logger.info(f"[{self.role}] Depositing to {self.venue}...")
        await asyncio.sleep(1.0)
        artifact.status = f"published_on_{self.venue}"
        logger.info(f"Success: DOI/ID minted for {self.venue}.")