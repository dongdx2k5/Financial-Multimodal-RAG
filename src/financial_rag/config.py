from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    # Google Gemini
    google_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"

    # Legacy OpenAI (optional, kept for backward compatibility)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 5

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "financial_rag.db"


settings = Settings()
