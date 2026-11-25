from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 5

    # Local OCR & Table Structure
    ocr_enabled: bool = False
    ocr_language: str = "vie+eng"
    tessdata_prefix: str | None = None
    tessdata_best_prefix: str | None = None
    ocr_backend: Literal["tesseract", "vietocr", "colab_gpu"] = "tesseract"
    ocr_fallback_to_tesseract: bool = True
    vietocr_weights: str | None = None
    ocr_detector_path: str | None = None
    table_backend: Literal["none", "slanet_plus"] = "none"
    table_model_path: str | None = None

    # Document Layout Analysis (separated from OCR_DETECTOR_PATH)
    layout_backend: Literal["none", "heuristic", "yolo"] = "none"
    layout_model_path: str | None = None
    chart_extraction_enabled: bool = True

    # Remote Google Colab T4+ GPU Inference API (local app keeps Web/SQLite DB)
    colab_gpu_url: str | None = None
    colab_api_token: str | None = None
    colab_timeout: float = 120.0
    colab_ocr_enabled: bool = False
    colab_embeddings_enabled: bool = False
    colab_allow_http_local: bool = True

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()
