from PIL import Image
from app.dataset_loader import iter_memes


def _fake_rows():
    for i in range(5):
        yield {"image": Image.new("RGB", (8, 8)), "id": f"row-{i}"}


def test_iter_memes_applies_limit():
    items = list(iter_memes(limit=3, rows=_fake_rows()))
    assert len(items) == 3


def test_iter_memes_zero_limit_means_all():
    items = list(iter_memes(limit=0, rows=_fake_rows()))
    assert len(items) == 5


def test_iter_memes_yields_id_image_metadata():
    meme_id, image, metadata = next(iter(iter_memes(limit=1, rows=_fake_rows())))
    assert isinstance(meme_id, str)
    assert isinstance(image, Image.Image)
    assert "image_ref" in metadata
