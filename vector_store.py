import json
import math
from typing import Any


# Default file created by build_vector_store.py.
DEFAULT_STORE_PATH = "vector_store.json"


def load_vector_store(path: str = DEFAULT_STORE_PATH) -> dict[str, Any]:
    """Load the local JSON vector store into memory."""
    with open(path, "r") as f:
        return json.load(f)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Measure how similar two embedding vectors are.

    Result range:
    - 1.0 means same direction, very similar.
    - 0.0 means unrelated/orthogonal.
    - Negative values mean opposite direction.

    This helper is not needed to build embeddings, but it is the core math you
    would use later to search `vector_store.json`.
    """
    # Vectors must exist and have the same number of dimensions.
    if not left or not right or len(left) != len(right):
        return 0.0

    # Dot product measures directional alignment.
    dot = sum(a * b for a, b in zip(left, right, strict=True))

    # Norms measure vector length. Cosine similarity divides by both lengths so
    # long chunks do not win just because their vectors have larger magnitude.
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
