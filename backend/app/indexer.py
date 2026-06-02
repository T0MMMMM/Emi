from typing import Callable, Iterable
from PIL import Image
from app.dataset_loader import iter_memes
from app.vector_store import FaissStore


def build_index(
    store: FaissStore,
    embedder,
    ocr: Callable[[Image.Image], str],
    rows: Iterable[dict] | None = None,
    limit: int = 0,
) -> int:
    """Indexe les memes : embedding image + OCR -> store. Renvoie le nb indexé.

    Les images illisibles sont ignorées (log via print) pour ne pas tout casser.
    """
    count = 0
    for meme_id, image, metadata in iter_memes(limit=limit, rows=rows):
        try:
            vector = embedder.embed_image(image)
            metadata = {**metadata, "ocr_text": ocr(image)}
            store.add([meme_id], [vector], [metadata])
            count += 1
        except Exception as exc:  # image corrompue / erreur d'encodage
            print(f"[indexer] meme {meme_id} ignoré : {exc}")
    return count
