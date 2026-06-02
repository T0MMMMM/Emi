from app.indexer import build_index
from app.vector_store import FaissStore


def test_build_index_populates_store(fake_embedder, color_image):
    rows = [
        {"image": color_image((255, 0, 0)), "id": "red"},
        {"image": color_image((0, 255, 0)), "id": "green"},
    ]
    store = FaissStore(dim=4)
    count = build_index(
        store=store,
        embedder=fake_embedder,
        ocr=lambda img: "",
        rows=rows,
        limit=0,
    )
    assert count == 2
    assert store.index.ntotal == 2
    assert store.metadatas[0]["image_ref"].endswith("#red")
