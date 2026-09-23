from typing import Any

from financial_rag.config.settings import Settings
from financial_rag.ingestion.models import DocumentElement
from financial_rag.reasoning.llm_client import UnifiedLLMClient

TABLE_SUMMARIZE_PROMPT = """You are an expert financial analyst. Analyze the following HTML table extracted from a financial document.
Describe clearly:
1. The table title / main topic and what each column/row represents.
2. Key financial metrics, numbers, percentages, and values.
3. Observed trends, comparisons across quarters/years, and financial highlights.
Keep the summary informative and concise for semantic search retrieval.

HTML Table:
{table_html}
"""

IMAGE_SUMMARIZE_PROMPT = """You are an expert multimodal financial analyst.
Examine this image/figure extracted from a financial report or research paper.
Provide a detailed description including:
1. Type of visual (e.g., bar chart, line graph, flow chart, system architecture, balance sheet visual).
2. Axes labels, legend items, categories, and numeric data points.
3. Key trends, growth/decline patterns, key takeaways, and conclusions.
Format the summary clearly for indexing and retrieval.
"""


class MultimodalSummarizer:
    """Generates semantic summaries for text, tables, and images using Unified LLM (Gemini or LLaMA)."""

    def __init__(self, settings: Settings, llm_client: UnifiedLLMClient | None = None):
        self.settings = settings
        try:
            self.llm_client: UnifiedLLMClient | None = llm_client or UnifiedLLMClient(settings)
        except Exception:
            self.llm_client = None

    @property
    def genai_client(self) -> Any:
        return self.llm_client.genai_client if self.llm_client else None

    def summarize_element(self, element: DocumentElement) -> str:
        modality = element.modality.value if hasattr(element.modality, "value") else str(element.modality).lower()
        if modality == "image":
            return self.summarize_image(element.raw_content or "", element.metadata)
        elif modality == "table":
            return self.summarize_table(element.raw_content or element.content)
        else:
            return self.summarize_text(element.raw_content or element.content)

    def summarize_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        # For short text chunks, direct text is already ideal for vector search
        if len(text) <= 500 or not self.llm_client:
            return text

        try:
            response = self.llm_client.generate_text(
                prompt=f"Provide a concise summary of the key facts, numbers, and points in this financial text for semantic search:\n\n{text[:2000]}",
                system_prompt="You are a concise financial summarizer for document indexing.",
                temperature=0.2,
            )
            return response or text
        except Exception:
            return text

    def summarize_table(self, table_html: str) -> str:
        if not table_html.strip():
            return ""
        if not self.llm_client:
            # Fallback when no API client is configured
            return f"Financial Table: {table_html[:400]}"

        try:
            prompt = TABLE_SUMMARIZE_PROMPT.format(table_html=table_html[:4000])
            response = self.llm_client.generate_text(prompt=prompt, temperature=0.2)
            return response or f"Financial Table Summary: {table_html[:300]}"
        except Exception as e:
            return f"Financial Table Data: {table_html[:400]} (Summarization fallback: {e})"

    def summarize_image(self, image_base64: str, metadata: dict[str, Any] | None = None) -> str:
        if not image_base64.strip():
            return "[Empty Image]"
        if not self.llm_client:
            return f"[Financial Visual/Chart: {metadata.get('image_name', 'image') if metadata else 'image'}]"

        try:
            mime_type = metadata.get("mime_type", "image/png") if metadata else "image/png"
            images = [{
                "image_base64": image_base64,
                "mime_type": mime_type,
            }]
            response = self.llm_client.generate_multimodal(
                prompt=IMAGE_SUMMARIZE_PROMPT,
                images=images,
                system_prompt="You are an expert multimodal financial analyst.",
                temperature=0.2,
            )
            return response or "[Financial Chart visual description unavailable]"
        except Exception as e:
            return f"[Financial Image: {metadata.get('image_name', 'extracted image') if metadata else 'image'} - Fallback: {e}]"
