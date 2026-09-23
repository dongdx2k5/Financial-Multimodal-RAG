"""Tests for UnifiedLLMClient and LLaMA API integration."""

from unittest.mock import MagicMock, patch
import pytest

from financial_rag.config.settings import Settings
from financial_rag.reasoning.llm_client import UnifiedLLMClient
from financial_rag.reasoning.financial_qa import FinancialQAEngine
from financial_rag.reasoning.pot_reasoning import ProgramOfThoughtEngine


class TestUnifiedLLMClientInit:
    def test_init_gemini_missing_key(self) -> None:
        settings = Settings(llm_provider="gemini", gemini_api_key=None, google_api_key=None, openai_api_key=None)
        with pytest.raises(ValueError, match="GEMINI_API_KEY chưa được cấu hình"):
            UnifiedLLMClient(settings)

    def test_init_llama_missing_key(self) -> None:
        settings = Settings(llm_provider="llama", llama_api_key=None, openai_api_key=None)
        with pytest.raises(ValueError, match="LLAMA_API_KEY.*chưa được cấu hình"):
            UnifiedLLMClient(settings)

    def test_init_llama_success(self) -> None:
        settings = Settings(
            llm_provider="llama",
            llama_api_key="gsk_test_key_123",
            llama_base_url="https://api.groq.com/openai/v1",
            llama_chat_model="llama-3.3-70b-versatile",
        )
        client = UnifiedLLMClient(settings)
        assert client.is_llama is True
        assert client.openai_client is not None
        assert str(client.openai_client.base_url) == "https://api.groq.com/openai/v1/"


class TestUnifiedLLMClientGeneration:
    def test_llama_text_generation(self) -> None:
        settings = Settings(
            llm_provider="llama",
            llama_api_key="test-key",
            llama_chat_model="llama-3.3-70b-versatile",
        )
        client = UnifiedLLMClient(settings)

        # Mock OpenAI chat completion
        mock_choice = MagicMock()
        mock_choice.message.content = "Revenue grew by 15.4% in 2021."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client.openai_client.chat.completions.create = MagicMock(return_value=mock_response)

        result = client.generate_text(
            prompt="What is the revenue growth?",
            system_prompt="You are a financial analyst.",
            temperature=0.0,
        )

        assert result == "Revenue grew by 15.4% in 2021."
        call_args = client.openai_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "llama-3.3-70b-versatile"
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["role"] == "user"

    def test_llama_multimodal_generation(self) -> None:
        settings = Settings(
            llm_provider="llama",
            llama_api_key="test-key",
            llama_chat_model="llama-3.2-11b-vision-preview",
        )
        client = UnifiedLLMClient(settings)

        mock_choice = MagicMock()
        mock_choice.message.content = "The chart displays an upward trend in EBITDA from Q1 to Q4."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client.openai_client.chat.completions.create = MagicMock(return_value=mock_response)

        images = [{
            "image_base64": "aW1hZ2VkYXRh",
            "mime_type": "image/png",
        }]

        result = client.generate_multimodal(
            prompt="Describe the chart trends.",
            images=images,
            system_prompt="Analyze this visual.",
        )

        assert "EBITDA" in result
        call_args = client.openai_client.chat.completions.create.call_args[1]
        user_message = call_args["messages"][1]
        # Verify image_url and text content format for LLaMA Vision
        assert isinstance(user_message["content"], list)
        assert user_message["content"][0]["type"] == "image_url"
        assert "data:image/png;base64,aW1hZ2VkYXRh" in user_message["content"][0]["image_url"]["url"]
        assert user_message["content"][1]["type"] == "text"


class TestLlamaPoTAndQAIntegration:
    def test_pot_with_llama(self) -> None:
        settings = Settings(
            llm_provider="llama",
            llama_api_key="test-key",
            llama_chat_model="llama-3.3-70b-versatile",
        )
        engine = ProgramOfThoughtEngine(settings)

        # Mock OpenAI code generation
        mock_choice = MagicMock()
        mock_choice.message.content = "```python\nrevenue_2020 = 1000.0\nrevenue_2021 = 1200.0\nans = round((revenue_2021 - revenue_2020) / revenue_2020 * 100, 2)\n```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        engine.llm_client.openai_client.chat.completions.create = MagicMock(return_value=mock_response)

        result = engine.solve(
            question="What was the revenue growth from 2020 to 2021?",
            context="Revenue 2020: $1,000M, Revenue 2021: $1,200M",
        )

        assert result["success"] is True
        assert result["answer"] == 20.0
        assert "ans = " in result["generated_code"]

    def test_qa_engine_with_llama(self) -> None:
        settings = Settings(
            llm_provider="llama",
            llama_api_key="test-key",
            llama_chat_model="llama-3.3-70b-versatile",
        )
        vector_store = MagicMock()
        vector_store.search.return_value = [
            {
                "doc_id": "doc1",
                "modality": "text",
                "content": "Net profit was $45M in 2022.",
                "raw_content": "Net profit was $45M in 2022.",
                "metadata": {"source": "sec_filing.pdf", "page": 4},
            }
        ]

        qa_engine = FinancialQAEngine(settings, vector_store)

        mock_choice = MagicMock()
        mock_choice.message.content = "Net profit in 2022 was $45M according to page 4 of the filing."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        qa_engine.llm_client.openai_client.chat.completions.create = MagicMock(return_value=mock_response)

        res = qa_engine.answer("What was the net profit in 2022?")
        assert "$45M" in res["answer"]
        assert len(res["contexts"]) == 1
