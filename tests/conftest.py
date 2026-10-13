from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Ensure tests run completely offline by default, without hitting live APIs."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")


@pytest.fixture
def mock_embedder():
    """Mock Gemini embedder providing deterministic vectors for testing."""
    embedder = MagicMock()
    # Return 8-dimensional dummy embeddings
    embedder.embed_documents.side_effect = lambda texts: np.ones((len(texts), 8), dtype=np.float32)
    embedder.embed_query.return_value = np.ones(8, dtype=np.float32)
    return embedder


@pytest.fixture
def mock_llm():
    """Mock Gemini LLM providing predictable answers for testing."""
    llm = MagicMock()
    llm.generate_answer.return_value = (
        "Theo báo cáo tài chính, Total Revenue năm 2024 là $245.1 billion [report_en.txt, page 1]."
    )
    return llm
