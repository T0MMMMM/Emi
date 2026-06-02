from typing import Iterable, Iterator
from PIL import Image
from app.config import Settings


def _hf_rows(settings: Settings) -> Iterable[dict]:
    """Lit le dataset Hugging Face en streaming (téléchargement réel)."""
    from datasets import load_dataset

    return load_dataset(settings.dataset_id, split="train", streaming=True)


def iter_memes(
    limit: int = 0,
    rows: Iterable[dict] | None = None,
    settings: Settings | None = None,
) -> Iterator[tuple[str, Image.Image, dict]]:
    """Itère sur les memes en (id, image, metadata).

    `rows` permet d'injecter une source (tests). Sinon, lit le dataset HF.
    `limit=0` => tout ; `limit>0` => sous-ensemble.
    """
    settings = settings or Settings()
    source = rows if rows is not None else _hf_rows(settings)
    for i, row in enumerate(source):
        if limit and i >= limit:
            break
        image = row["image"]
        meme_id = str(row.get("id", i))
        metadata = {"image_ref": f"{settings.dataset_id}#{meme_id}"}
        yield meme_id, image, metadata
