from PIL import Image, ImageDraw
from app.ocr import extract_text


def _image_with_text(text: str) -> Image.Image:
    img = Image.new("RGB", (320, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), text, fill="black")
    return img


def test_extract_reads_written_text():
    img = _image_with_text("HELLO")
    assert "HELLO" in extract_text(img).upper()


def test_extract_blank_image_returns_empty():
    img = Image.new("RGB", (320, 80), color="white")
    assert extract_text(img).strip() == ""
