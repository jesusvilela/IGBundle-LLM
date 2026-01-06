
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from enum import Enum

class ReviewDecision(Enum):
    ACCEPT = "ACCEPT"
    MINOR_REVISION = "MINOR_REVISION"
    MAJOR_REVISION = "MAJOR_REVISION"
    REJECT = "REJECT"

@dataclass
class ManuscriptPackage:
    title: str
    content: str  # Markdown or LaTeX source
    authors: List[str]
    figures: Dict[str, str] = field(default_factory=dict) # path or description
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    version: int = 1

@dataclass
class ReviewIssue:
    severity: str # "major", "minor"
    description: str
    location: Optional[str] = None
    category: str = "general" # correctness, clarity, ethics, etc.

@dataclass
class ReviewResult:
    reviewer_name: str
    role: str
    scores: Dict[str, float] # 0.0 to 10.0
    summary: str
    issues: List[ReviewIssue]
    confidence: float

@dataclass
class ConsolidatedReview:
    decision: ReviewDecision
    synthesis: str
    required_actions: List[str]
    reviews: List[ReviewResult]

@dataclass
class PublicationArtifact:
    pdf_path: str
    source_archive_path: str
    metadata_json_path: str
    target_venue: str
    status: str = "pending"
