from fastapi.testclient import TestClient
from app.api import create_app


class StubEngine:
    def search_text(self, query, k):
        return {"found": {"id": "x", "score": 0.99, "metadata": {"image_ref": "x"}},
                "other_versions": [], "similar": []}

    def search_image(self, image, k):
        return {"found": {"id": "y", "score": 0.98, "metadata": {"image_ref": "y"}},
                "other_versions": [], "similar": []}


def _client():
    app = create_app(engine=StubEngine())
    return TestClient(app)


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_text():
    resp = _client().post("/search/text", json={"query": "pink hamster", "k": 10})
    assert resp.status_code == 200
    assert resp.json()["found"]["id"] == "x"


def test_search_text_requires_query():
    resp = _client().post("/search/text", json={})
    assert resp.status_code == 422


def test_search_image():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    resp = _client().post("/search/image", files={"file": ("m.png", buf, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["found"]["id"] == "y"
