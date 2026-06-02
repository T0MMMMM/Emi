from PIL import Image


def categorize(results: list[dict], version_threshold: float) -> dict:
    """Classe les résultats triés par score décroissant en found / versions / similar."""
    if not results:
        return {"found": None, "other_versions": [], "similar": []}
    found, *rest = results
    other_versions = [r for r in rest if r["score"] >= version_threshold]
    similar = [r for r in rest if r["score"] < version_threshold]
    return {"found": found, "other_versions": other_versions, "similar": similar}


class SearchEngine:
    """Recherche par texte ou image -> résultats catégorisés."""

    def __init__(self, store, embedder, version_threshold: float = 0.90):
        self._store = store
        self._embedder = embedder
        self._version_threshold = version_threshold

    def search_text(self, query: str, k: int) -> dict:
        vector = self._embedder.embed_text(query)
        return categorize(self._store.search(vector, k), self._version_threshold)

    def search_image(self, image: Image.Image, k: int) -> dict:
        vector = self._embedder.embed_image(image)
        return categorize(self._store.search(vector, k), self._version_threshold)
