from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "development"
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/vectorstore")
    collection_name: str = "financial_reports"

    # LLM Provider: "gemini" (default) or "llama" / "openai"
    llm_provider: str = "gemini"

    # Gemini settings
    embedding_model: str = "gemini-embedding-001"
    chat_model: str = "gemini-3.6-flash"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None

    # LLaMA / OpenAI-compatible API settings (Groq, Together AI, OpenRouter, vLLM, Ollama)
    llama_base_url: str = "https://api.groq.com/openai/v1"
    llama_api_key: str | None = None
    llama_chat_model: str = "llama-3.3-70b-versatile"
    llama_embedding_model: str | None = None

    # ─────────────────────────────────────────────
    # GraphRAG settings
    # ─────────────────────────────────────────────
    # Directory to persist the Knowledge Graph (JSON + GEXF)
    graph_dir: Path = Path("data/graph")

    # Maximum number of hops when expanding a subgraph from seed nodes
    graph_max_hops: int = 2

    # Enable GraphRAG mode (graph traversal + vector hybrid retrieval)
    use_graph_rag: bool = True

    # HuggingFace datasets local cache directory
    hf_cache_dir: Path = Path("data/raw/datasets/huggingface")

    # Maximum number of entity-extraction LLM calls during graph build
    # Set to None for unlimited (warning: expensive for large datasets)
    graph_entity_extract_limit: int | None = 200

    @property
    def api_key(self) -> str | None:
        """Returns the default Google/Gemini API key."""
        return self.gemini_api_key or self.google_api_key or self.openai_api_key

    @property
    def is_llama(self) -> bool:
        """Checks if current LLM provider is LLaMA or OpenAI-compatible."""
        return self.llm_provider.strip().lower() in (
            "llama",
            "openai",
            "openai_compatible",
            "groq",
            "together",
            "openrouter",
            "ollama",
        )

    @property
    def active_api_key(self) -> str | None:
        """Returns the API key for the active provider."""
        if self.is_llama:
            return self.llama_api_key or self.openai_api_key
        return self.api_key

    @property
    def active_chat_model(self) -> str:
        """Returns the model name for the active provider."""
        if self.is_llama:
            return self.llama_chat_model
        return self.chat_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
