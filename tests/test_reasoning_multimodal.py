from unittest.mock import MagicMock
from financial_rag.config.settings import Settings
from financial_rag.reasoning.financial_qa import FinancialQAEngine


def test_multimodal_qa_prompt_construction() -> None:
    settings = Settings(gemini_api_key="test-mock-key")
    vector_store = MagicMock()

    # Mock search returning text, table, and image
    vector_store.search.return_value = [
        {
            "doc_id": "txt_1",
            "modality": "text",
            "content": "Summary of revenue text",
            "raw_content": "Full revenue text in Q3 was $500M.",
            "metadata": {"source": "report.pdf", "page": 1},
        },
        {
            "doc_id": "tbl_1",
            "modality": "table",
            "content": "Summary of Q3 profit table",
            "raw_content": "<table><tr><td>Profit</td><td>$50M</td></tr></table>",
            "metadata": {"source": "report.pdf", "page": 2},
        },
        {
            "doc_id": "img_1",
            "modality": "image",
            "content": "Bar chart of quarterly growth",
            "raw_content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "metadata": {"source": "report.pdf", "page": 3, "image_path": "data/images/chart.png"},
        },
    ]

    qa_engine = FinancialQAEngine(settings, vector_store)
    
    # Mock genai client generate_content
    mock_response = MagicMock()
    mock_response.text = "Q3 revenue was $500M with $50M profit as shown in Table 1 and Chart 1."
    qa_engine.genai_client.models.generate_content = MagicMock(return_value=mock_response)

    result = qa_engine.answer("What was the financial performance in Q3?", limit=3)

    assert result["answer"] == mock_response.text
    assert len(result["contexts"]) == 3
    assert len(result["retrieved_images"]) == 1
    assert len(result["retrieved_tables"]) == 1
    assert result["retrieved_images"][0]["doc_id"] == "img_1"
    assert result["retrieved_tables"][0]["doc_id"] == "tbl_1"
    assert "<table>" in result["retrieved_tables"][0]["table_html"]
