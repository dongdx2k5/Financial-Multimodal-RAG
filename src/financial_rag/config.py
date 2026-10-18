"""Configuration module for Financial RAG."""

from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base workspace directory
    base_dir: Path = Path(__file__).resolve().parent.parent.parent

    # Data paths
    data_dir: Path = Path("data")
    manifest_path: Path = Path("data/manifest.csv")
    database_path: Path = Path("data/structured/financial.db")
    metrics_seed_path: Path = Path("data/structured/metrics_seed.csv")
    faiss_index_path: Path = Path("data/index/faiss.index")
    chunk_ids_path: Path = Path("data/index/chunk_ids.json")
    eval_questions_path: Path = Path("data/eval/questions.csv")

    # LLM Settings
    llm_backend: Literal["qwen_local", "mock", "openai"] = "mock"
    llm_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_device: str = "cuda"
    llm_fp16: bool = True
    llm_max_new_tokens: int = 512
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Embedding & Retrieval Settings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    top_k_chunks: int = 4
    dense_weight: float = 0.6
    fts_weight: float = 0.4
    relevance_threshold: float = 0.35

    # Gradio UI Settings
    gradio_share: bool = False
    gradio_server_port: int = 7860

    def resolve_path(self, path: Path) -> Path:
        """Resolve relative path against base directory if not absolute."""
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()


settings = Settings()
