from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "development"
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/vectorstore")
    collection_name: str = "financial_reports"
    embedding_model: str = "gemini-embedding-001"
    chat_model: str = "gemini-3.6-flash"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None

    @property
    def api_key(self) -> str | None:
        return self.gemini_api_key or self.google_api_key or self.openai_api_key



@lru_cache
def get_settings() -> Settings:
    return Settings()

