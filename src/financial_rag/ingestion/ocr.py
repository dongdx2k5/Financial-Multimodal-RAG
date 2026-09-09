from pathlib import Path

import pytesseract
from PIL import Image

from financial_rag.ingestion.models import DocumentElement


def load_image_with_ocr(path: Path, language: str = "eng") -> DocumentElement:
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang=language)
    return DocumentElement(text.strip(), "image_ocr", path, metadata={"ocr_language": language})

