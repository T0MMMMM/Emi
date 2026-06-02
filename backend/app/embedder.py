from functools import cached_property

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import Settings


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise L2 un vecteur en float32. Vecteur nul -> reste nul (pas de NaN)."""
    vec = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype("float32")


class CLIPEmbedder:
    """Encode images et textes dans le MÊME espace CLIP 512-D (cosinus).

    - images : modèle CLIP visuel
    - texte  : modèle CLIP texte multilingue, aligné sur le même espace
    Les modèles sont chargés à la première utilisation (lazy).
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()

    @cached_property
    def _image_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._settings.image_model)

    @cached_property
    def _text_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._settings.text_model)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        vec = self._image_model.encode(image.convert("RGB"))
        return _normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        vec = self._text_model.encode(text)
        return _normalize(vec)
