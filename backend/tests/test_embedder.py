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


import pytest
from PIL import Image
from app.embedder import CLIPEmbedder


@pytest.fixture(scope="module")
def embedder():
    return CLIPEmbedder()


@pytest.mark.integration
def test_embed_text_shape_and_norm(embedder):
    vec = embedder.embed_text("a pink hamster")
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


@pytest.mark.integration
def test_embed_image_shape_and_norm(embedder):
    img = Image.new("RGB", (64, 64), color=(255, 0, 128))
    vec = embedder.embed_image(img)
    assert vec.shape == (512,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


@pytest.mark.integration
def test_text_and_image_share_space(embedder):
    # une description doit être plus proche de l'image correspondante que d'une autre
    img = Image.new("RGB", (64, 64), color=(255, 0, 128))
    img_vec = embedder.embed_image(img)
    close = float(np.dot(embedder.embed_text("a pink square"), img_vec))
    far = float(np.dot(embedder.embed_text("a green forest landscape"), img_vec))
    assert close > far
