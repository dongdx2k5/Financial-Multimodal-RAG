"""LLM package for Financial RAG."""

from financial_rag.llm.prompt import (
    SYSTEM_PROMPT,
    SLOT_EXTRACTION_PROMPT,
    VERBALIZATION_PROMPT,
    TEXT_EXPLANATION_PROMPT,
)
from financial_rag.llm.model import LLMBackend

__all__ = [
    "SYSTEM_PROMPT",
    "SLOT_EXTRACTION_PROMPT",
    "VERBALIZATION_PROMPT",
    "TEXT_EXPLANATION_PROMPT",
    "LLMBackend",
]
