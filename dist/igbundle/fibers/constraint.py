import re
from typing import List, Dict, Any

class ConstraintExtractor:
    """
    Extracts semantic constraints from a prompt.
    v1: Heuristic/Keyword based.
    """
    def __init__(self):
        self.geo_terms = [
            "riemannian", "manifold", "geometry", "curvature", "metric", "topology", 
            "fiber", "bundle", "geodesic", "vector field", "tangent space"
        ]
        self.ai_terms = [
            "artificial intelligence", "ai", "neural network", "deep learning", "model", 
            "representation", "latent space", "embedding"
        ]

    def extract(self, prompt: str) -> List[str]:
        prompt_lower = prompt.lower()
        constraints = []

        # Check for Geometric Constraint
        if any(term in prompt_lower for term in self.geo_terms):
            constraints.append("geometric_explanation")
            
        # Check for AI Constraint (usually implicit, but good to track)
        if any(term in prompt_lower for term in self.ai_terms):
            constraints.append("ai_context")
            
        # Check for "How/Why" (Explanatory depth)
        if "how" in prompt_lower or "why" in prompt_lower or "explain" in prompt_lower:
             constraints.append("explanatory_depth")
             
        return constraints

class ConstraintScorer:
    """
    Scores a response against a list of constraints.
    """
    def __init__(self):
        # We map constraints to required evidential terms
        self.evidence_map = {
            "geometric_explanation": [
                "manifold", "curvature", "metric", "space", "geometry", "structure", 
                "continuous", "map", "distance", "invariant"
            ],
            "ai_context": [
               "agent", "model", "data", "learning", "reward", "representation", "function", 
               "optimization", "parameter"
            ],
            "explanatory_depth": [
                "because", "therefore", "implies", "reason", "mechanism", "structure"
            ]
        }

    def score(self, response: str, constraints: List[str]) -> Dict[str, float]:
        response_lower = response.lower()
        scores = {}
        
        for c in constraints:
            if c not in self.evidence_map:
                scores[c] = 1.0 # Unknown constraint, assume satisfied or ignore
                continue
                
            evidence_terms = self.evidence_map[c]
            hits = sum(1 for term in evidence_terms if term in response_lower)
            
            # Simple soft saturation score
            # 0 hits = 0.0
            # 1 hit = 0.5
            # 2 hits = 0.8
            # 3+ hits = 1.0
            
            if hits == 0:
                scores[c] = 0.0
            elif hits == 1:
                scores[c] = 0.5
            elif hits == 2:
                scores[c] = 0.8
            else:
                scores[c] = 1.0
                
        return scores
