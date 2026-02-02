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
        
        # Tokenize prompt for fuzzy matching
        prompt_tokens = prompt_lower.split()

        # Check for Geometric Constraint (Fuzzy)
        # Check exact substring first
        geo_hit = any(term in prompt_lower for term in self.geo_terms)
        
        # If no exact hit, check fuzzy
        if not geo_hit:
            import difflib
            for token in prompt_tokens:
                # Check against geo terms
                matches = difflib.get_close_matches(token, self.geo_terms, n=1, cutoff=0.8)
                if matches:
                    geo_hit = True
                    break
        
        if geo_hit:
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
                "manifold", "curvature", "metric", "geometry", "geodesic", 
                "continuous", "map", "distance", "invariant", "tensor", "topology"
            ],
            "ai_context": [
               "agent", "model", "data", "learning", "reward", "representation", "function", 
               "optimization", "parameter"
            ],
            "explanatory_depth": [
                "because", "therefore", "implies", "reason", "mechanism", "structure"
            ]
        }

    def check_repetition(self, text: str) -> float:
        """
        Detects repetition loops. Returns penalty factor (0.0 = Bad Loop, 1.0 = Clean).
        """
        if len(text) < 100: return 1.0
        
        # 1. Compression Ratio (The "Entropy Check")
        # Repeating text compresses extremely well.
        import zlib
        compressed = zlib.compress(text.encode('utf-8'))
        ratio = len(text) / len(compressed)
        
        # Normal prose is ~1.5-2.0. Loops are > 3.0.
        if ratio > 3.5:
             return 0.0 # Severe Loop
        if ratio > 2.8:
             return 0.2 # Probable Loop
             
        # 2. Semantic Stagnation (Sentence Overlap)
        # Check if the last sentence is just a reshuffle of the previous one.
        # "Set of elements is universe. Universe is set of elements."
        
        # Split by typical sentence delimiters
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s for s in sentences if len(s.split()) > 5] # Filter short ones
        
        if len(sentences) >= 2:
            s_last = set(sentences[-1].lower().split())
            s_prev = set(sentences[-2].lower().split())
            
            if len(s_last) > 0 and len(s_prev) > 0:
                # Jaccard index
                intersection = len(s_last.intersection(s_prev))
                union = len(s_last.union(s_prev))
                overlap = intersection / union
                
                if overlap > 0.85: # Tautology detected
                    return 0.0
                    
        return 1.0

    def score(self, response: str, constraints: List[str]) -> Dict[str, float]:
        response_lower = response.lower()
        scores = {}
        
        # 0. Global Loop Check
        loop_factor = self.check_repetition(response)
        
        for c in constraints:
            if c not in self.evidence_map:
                scores[c] = 1.0 * loop_factor 
                continue
                
            evidence_terms = self.evidence_map[c]
            hits = sum(1 for term in evidence_terms if term in response_lower)
            
            # Simple soft saturation score
            if hits == 0:
                s = 0.0
            elif hits == 1:
                s = 0.5
            elif hits == 2:
                s = 0.8
            else:
                s = 1.0
            
            scores[c] = s * loop_factor
                
        return scores
