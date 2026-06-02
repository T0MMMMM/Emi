import numpy as np
from app.vector_store import FaissStore


def _unit(*values) -> np.ndarray:
    v = np.array(values, dtype="float32")
    return v / np.linalg.norm(v)


def test_search_returns_nearest_first():
    store = FaissStore(dim=2)
    store.add(
        ids=["a", "b"],
        vectors=[_unit(1, 0), _unit(0, 1)],
        metadatas=[{"image_ref": "a.jpg"}, {"image_ref": "b.jpg"}],
    )
    results = store.search(_unit(1, 0), k=2)
    assert results[0]["id"] == "a"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["metadata"]["image_ref"] == "a.jpg"


def test_save_and_load_roundtrip(tmp_path):
    store = FaissStore(dim=2)
    store.add(["a"], [_unit(1, 0)], [{"image_ref": "a.jpg", "ocr_text": "hi"}])
    store.save(str(tmp_path))

    loaded = FaissStore.load(str(tmp_path))
    results = loaded.search(_unit(1, 0), k=1)
    assert results[0]["id"] == "a"
    assert results[0]["metadata"]["ocr_text"] == "hi"


def test_empty_store_search_returns_empty():
    store = FaissStore(dim=2)
    assert store.search(_unit(1, 0), k=5) == []
