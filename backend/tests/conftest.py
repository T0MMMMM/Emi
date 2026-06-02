import numpy as np
import pytest
from PIL import Image


class FakeEmbedder:
    """Embedder déterministe sans modèle : couleur moyenne -> vecteur normalisé."""

    def embed_image(self, image: Image.Image) -> np.ndarray:
        r, g, b = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0)
        v = np.array([r, g, b, 1.0], dtype="float32")
        return v / np.linalg.norm(v)

    def embed_text(self, text: str) -> np.ndarray:
        # mappe quelques mots-clés vers une couleur pour des tests lisibles
        table = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
        r, g, b = table.get(text.strip().lower(), (0, 0, 0))
        v = np.array([r, g, b, 1.0], dtype="float32")
        return v / np.linalg.norm(v)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def color_image():
    def _make(rgb):
        return Image.new("RGB", (8, 8), color=rgb)
    return _make
