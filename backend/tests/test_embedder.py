import numpy as np
from app.embedder import _normalize


def test_normalize_unit_norm():
    vec = np.array([3.0, 4.0], dtype="float32")
    out = _normalize(vec)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert out.dtype == np.float32


def test_normalize_zero_vector_is_safe():
    vec = np.zeros(4, dtype="float32")
    out = _normalize(vec)
    # pas de division par zéro : reste à zéro
    assert np.all(out == 0.0)
    assert out.dtype == np.float32
