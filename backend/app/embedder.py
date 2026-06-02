import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise L2 un vecteur en float32. Vecteur nul -> reste nul (pas de NaN)."""
    vec = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype("float32")
