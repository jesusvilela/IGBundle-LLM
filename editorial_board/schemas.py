from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ManuscriptPackage:
    source_path: str
    figures: List[str] = field(default_factory=list)
    bibliography_path: str = ""
    claimed_citations: List[str] = field(default_factory=list)
    data_repo_url: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReviewIssue:
    type: str  # "major", "minor"
    description: str
    location: str = ""

@dataclass
class ReviewBundle:
    reviewer_id: str
    scores: Dict[str, float]  # novelty, correctness, clarity, reproducibility, ethics, citation_hygiene
    major_issues: List[ReviewIssue] = field(default_factory=list)
    minor_issues: List[ReviewIssue] = field(default_factory=list)
    required_fixes: List[str] = field(default_factory=list)
    claim_checks: Dict[str, str] = field(default_factory=dict)  # claim -> status
    recommendation: str = ""  # Reject, Revise, Accept

@dataclass
class MetaReview:
    decision: str
    summary: str
    revision_plan: List[str] = field(default_factory=list)

@dataclass
class PublishPackage:
    final_pdf_path: str
    source_archive_path: str
    metadata_files: Dict[str, str] = field(default_factory=dict)
    disclosures: str = ""
    version: str = "1.0.0"
