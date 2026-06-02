import pytesseract
from PIL import Image
from app.config import Settings

_SETTINGS = Settings()


def extract_text(image: Image.Image, languages: str | None = None) -> str:
    """Extrait le texte écrit sur une image. Renvoie '' si rien / en cas d'erreur OCR."""
    langs = languages or _SETTINGS.ocr_languages
    try:
        return pytesseract.image_to_string(image.convert("RGB"), lang=langs).strip()
    except pytesseract.TesseractError:
        return ""
