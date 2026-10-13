from unittest.mock import MagicMock, patch

import numpy as np

from financial_rag.embeddings import GeminiEmbedder
from financial_rag.llm import GeminiLLM


def test_gemini_embedder_mocked():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb]
        mock_client.models.embed_content.return_value = mock_response

        embedder = GeminiEmbedder(api_key="test-key", model_name="gemini-embedding-001")

        # Test document embedding
        docs_vec = embedder.embed_documents(["Test chunk 1"])
        assert docs_vec.shape == (1, 3)
        assert np.allclose(docs_vec[0], [0.1, 0.2, 0.3])

        # Test query embedding
        query_vec = embedder.embed_query("Query")
        assert query_vec.shape == (3,)
        assert np.allclose(query_vec, [0.1, 0.2, 0.3])


def test_gemini_llm_mocked():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Doanh thu năm 2024 là 125,5 tỷ đồng [source.txt, trang 1]."
        mock_client.models.generate_content.return_value = mock_response

        llm = GeminiLLM(api_key="test-key", model_name="gemini-3.8-flash")
        answer = llm.generate_answer(
            question="Doanh thu năm 2024?",
            evidence_texts=["[source.txt, trang 1]: Doanh thu 125,5 tỷ"],
        )
        assert "125,5" in answer
