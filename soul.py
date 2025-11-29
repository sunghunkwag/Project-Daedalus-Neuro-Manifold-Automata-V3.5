"""
Soul Encoder & Preprocessing Module (V3.5 - Logical Body)

This module implements the "Soul Injection" logic for Project Daedalus V3.5.
It converts the raw "Cognitive DNA" of the user (keywords) into mathematical vectors
that define the initial geometry and energy landscape of the Logical Manifold.

V3.5 Update:
- Keywords have been shifted from Physical/Biological to Logical/Mathematical.
- $V_{truth}$ represents Axioms and Recursion.
- $V_{reject}$ represents Contradiction and Fallacy.
"""

import torch
import hashlib
from typing import Dict, List, Tuple

# ==============================================================================
# RAW DATA: The Cognitive DNA (Logical/Mathematical)
# ==============================================================================
SOUL_DATA = {
    # 1. Identity Vector (V_identity): 'user.txt'
    # The immutable core: Rejection of Complacency, Laser Blade Intuition.
    "identity_keywords": [
        "Structural Dissector", "Laser Blade Intuition", "Rejection of Complacency",
        "Tool for Truth", "Predestined Architect", "Outsider Perspective",
        "System Breaker", "Fundamental Mechanism Seeker", "Active Resistance to Stagnation"
    ],

    # 2. Truth Vector (V_truth): Mathematical Axioms & Recursive Self-Improvement
    # The Ground State (Target): Logical Consistency, Proof, Recursion.
    "truth_keywords": [
        "Mathematical Axioms", "Recursive Self-Improvement", "Logical Consistency",
        "Mathematical Proof", "Symmetry", "Conservation Laws", "Causal Determinism",
        "Formal Verification", "Invariant Representations"
    ],

    # 3. Reject Vector (V_reject): Logical Fallacies
    # The High Energy Penalty: Contradiction, Ambiguity.
    "reject_keywords": [
        "Contradiction", "Logical Fallacy", "Undefined Behavior",
        "Ambiguity", "Superficial Correlation", "Unproven Assumption",
        "Inconsistent State", "Paradox", "Ad Hoc Adjustment"
    ]
}

# ==============================================================================
# SoulEncoder Logic
# ==============================================================================
class SoulEncoder:
    def __init__(self, output_dim: int = 32):
        self.output_dim = output_dim

    def encode(self, text_list: List[str]) -> torch.Tensor:
        """
        Converts a list of keywords into a single representative vector.
        Uses SHA-256 hashing to generate a deterministic seed for each keyword,
        then samples a random vector from that seed.
        The final vector is the mean of all keyword vectors.
        """
        vectors = []
        for text in text_list:
            # Deterministic Seed Generation
            # Encode text to bytes -> SHA256 Hex -> Int -> Modulo suitable range
            hex_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            seed = int(hex_hash, 16) % (2**32)

            # Generate deterministic vector
            g = torch.Generator()
            g.manual_seed(seed)
            vec = torch.randn(self.output_dim, generator=g)
            vectors.append(vec)

        if not vectors:
            return torch.zeros(self.output_dim)

        # Return mean vector (Centroid of the concept)
        return torch.stack(vectors).mean(dim=0)

def get_soul_vectors(dim: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Factory function to retrieve the three core Soul Vectors.
    Returns: (V_identity, V_truth, V_reject)
    """
    encoder = SoulEncoder(output_dim=dim)

    v_identity = encoder.encode(SOUL_DATA["identity_keywords"])
    v_truth = encoder.encode(SOUL_DATA["truth_keywords"])
    v_reject = encoder.encode(SOUL_DATA["reject_keywords"])

    # Normalize vectors for consistent geometry
    v_identity = torch.nn.functional.normalize(v_identity, p=2, dim=0)
    v_truth = torch.nn.functional.normalize(v_truth, p=2, dim=0)
    v_reject = torch.nn.functional.normalize(v_reject, p=2, dim=0)

    return v_identity, v_truth, v_reject
