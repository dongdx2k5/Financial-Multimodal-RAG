from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    openai_api_key: str | None = None
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout: float = 180
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    ocr_enabled: bool = False
    ocr_language: str = "vie+eng"
    tessdata_prefix: str | None = None
    tessdata_best_prefix: str | None = None

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()
