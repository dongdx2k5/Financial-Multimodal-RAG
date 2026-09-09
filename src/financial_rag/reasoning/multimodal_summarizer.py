import base64
import time
from typing import Any

from google import genai
from google.genai import types

from financial_rag.config.settings import Settings
from financial_rag.ingestion.models import DocumentElement

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
    """Generates semantic summaries for text, tables, and images using Gemini Multimodal LLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        api_key = settings.api_key
        self.genai_client = genai.Client(api_key=api_key) if api_key else None

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
        if len(text) <= 500 or not self.genai_client:
            return text

        try:
            response = self._call_llm(
                prompt=f"Provide a concise summary of the key facts, numbers, and points in this financial text for semantic search:\n\n{text[:2000]}",
                system_prompt="You are a concise financial summarizer for document indexing.",
            )
            return response or text
        except Exception:
            return text

    def summarize_table(self, table_html: str) -> str:
        if not table_html.strip():
            return ""
        if not self.genai_client:
            # Fallback when no API client is configured
            return f"Financial Table: {table_html[:400]}"

        try:
            prompt = TABLE_SUMMARIZE_PROMPT.format(table_html=table_html[:4000])
            response = self._call_llm(prompt=prompt)
            return response or f"Financial Table Summary: {table_html[:300]}"
        except Exception as e:
            return f"Financial Table Data: {table_html[:400]} (Summarization fallback: {e})"

    def summarize_image(self, image_base64: str, metadata: dict[str, Any] | None = None) -> str:
        if not image_base64.strip():
            return "[Empty Image]"
        if not self.genai_client:
            return f"[Financial Visual/Chart: {metadata.get('image_name', 'image') if metadata else 'image'}]"

        try:
            image_bytes = base64.b64decode(image_base64)
            mime_type = metadata.get("mime_type", "image/png") if metadata else "image/png"
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            response = self.genai_client.models.generate_content(
                model=self.settings.chat_model,
                contents=[image_part, IMAGE_SUMMARIZE_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
            return response.text or "[Financial Chart visual description unavailable]"
        except Exception as e:
            return f"[Financial Image: {metadata.get('image_name', 'extracted image') if metadata else 'image'} - Fallback: {e}]"

    def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        if not self.genai_client:
            return ""
        for attempt in range(4):
            try:
                config = types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction=system_prompt,
                ) if system_prompt else types.GenerateContentConfig(temperature=0.2)

                response = self.genai_client.models.generate_content(
                    model=self.settings.chat_model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise e
        return ""
