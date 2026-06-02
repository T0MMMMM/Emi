from app.search import SearchEngine, categorize
from app.vector_store import FaissStore


def test_categorize_splits_found_versions_similar():
    results = [
        {"id": "a", "score": 0.99, "metadata": {}},
        {"id": "b", "score": 0.95, "metadata": {}},  # autre version
        {"id": "c", "score": 0.40, "metadata": {}},  # similaire
    ]
    out = categorize(results, version_threshold=0.90)
    assert out["found"]["id"] == "a"
    assert [r["id"] for r in out["other_versions"]] == ["b"]
    assert [r["id"] for r in out["similar"]] == ["c"]


def test_categorize_empty_results():
    out = categorize([], version_threshold=0.90)
    assert out == {"found": None, "other_versions": [], "similar": []}


def test_search_text_uses_text_embedding(fake_embedder, color_image):
    store = FaissStore(dim=4)
    store.add(["red"], [fake_embedder.embed_image(color_image((255, 0, 0)))], [{"image_ref": "r"}])
    store.add(["blue"], [fake_embedder.embed_image(color_image((0, 0, 255)))], [{"image_ref": "b"}])

    engine = SearchEngine(store=store, embedder=fake_embedder, version_threshold=0.90)
    out = engine.search_text("red", k=2)
    assert out["found"]["id"] == "red"


def test_search_image_uses_image_embedding(fake_embedder, color_image):
    store = FaissStore(dim=4)
    store.add(["green"], [fake_embedder.embed_image(color_image((0, 255, 0)))], [{"image_ref": "g"}])

    engine = SearchEngine(store=store, embedder=fake_embedder, version_threshold=0.90)
    out = engine.search_image(color_image((0, 255, 0)), k=1)
    assert out["found"]["id"] == "green"
