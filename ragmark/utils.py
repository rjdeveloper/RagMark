"""
RagMark — Utility functions.

Pure-Python helpers with zero external dependencies (no NumPy).
"""

import math
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns a value in [-1, 1].  Returns 0.0 when either vector has
    zero magnitude (graceful handling of zero-vectors).

    Parameters
    ----------
    a, b : list[float]
        Embedding vectors of the same dimensionality.

    Returns
    -------
    float
        Cosine similarity score.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have equal length, got {len(a)} and {len(b)}"
        )

    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0

    for ai, bi in zip(a, b):
        dot += ai * bi
        mag_a += ai * ai
        mag_b += bi * bi

    mag_a = math.sqrt(mag_a)
    mag_b = math.sqrt(mag_b)

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)
