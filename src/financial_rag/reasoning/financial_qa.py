import base64
from typing import Any

from google import genai
from google.genai import types

from financial_rag.config.settings import Settings
from financial_rag.retrieval.vector_store import FinancialVectorStore

SYSTEM_PROMPT = """You are an advanced Multimodal Financial Analyst AI.
You have access to retrieved contextual evidence including textual narratives, structured HTML tables, and visual charts/figures.

Guidelines:
1. Ground your answers strictly on the supplied context (texts, tables, and images).
2. When answering, cite specific metrics, numbers, percentages, dates, and trends from the tables and charts.
3. If an image or chart provides visual trends or architecture details, explain what the chart reveals.
4. If the evidence is insufficient or contradictory, explicitly state what is missing. Do not invent figures or assumptions.
5. Structure your answer with:
   - **Executive Summary / Conclusion**: Direct answer to the question.
   - **Supporting Evidence & Details**: Key figures, breakdown of tables, and visual chart observations.
   - **Sources & References**: Page numbers and file sources.
"""


class FinancialQAEngine:
    """Multimodal QA Engine: Uses Multimodal LLM to reason over text, tables, and images."""

    def __init__(self, settings: Settings, vector_store: FinancialVectorStore):
        self.settings = settings
        self.vector_store = vector_store
        api_key = settings.api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY chưa được cấu hình! Vui lòng điền GEMINI_API_KEY vào file '.env'."
            )
        self.genai_client = genai.Client(api_key=api_key)

    def answer(self, question: str, limit: int = 6) -> dict[str, Any]:
        contexts = self.vector_store.search(question, limit=limit)
        
        text_table_contexts = []
        retrieved_images: list[dict[str, Any]] = []
        retrieved_tables: list[dict[str, Any]] = []
        image_parts = []

        for idx, item in enumerate(contexts, start=1):
            modality = str(item.get("modality", "text")).lower()
            source = item.get("metadata", {}).get("source", "unknown")
            page = item.get("metadata", {}).get("page", 0)
            raw_content = item.get("raw_content", item.get("content", ""))

            if modality == "image":
                retrieved_images.append({
                    "doc_id": item.get("doc_id"),
                    "source": source,
                    "page": page,
                    "summary": item.get("content"),
                    "image_base64": raw_content,
                    "metadata": item.get("metadata", {}),
                })
                # Decode image for Gemini Vision
                try:
                    if raw_content and isinstance(raw_content, str):
                        img_bytes = base64.b64decode(raw_content)
                        mime = item.get("metadata", {}).get("meta_mime_type", "image/png")
                        image_parts.append(
                            types.Part.from_bytes(data=img_bytes, mime_type=mime)
                        )
                except Exception:
                    pass

            elif modality == "table":
                retrieved_tables.append({
                    "doc_id": item.get("doc_id"),
                    "source": source,
                    "page": page,
                    "table_html": raw_content,
                    "summary": item.get("content"),
                    "metadata": item.get("metadata", {}),
                })
                text_table_contexts.append(
                    f"[Context #{idx} - Structured Table | Source: {source}, Page: {page}]\n{raw_content}"
                )

            else:
                text_table_contexts.append(
                    f"[Context #{idx} - Text Narrative | Source: {source}, Page: {page}]\n{raw_content}"
                )

        combined_text_context = "\n\n" + ("\n\n---\n\n".join(text_table_contexts)) if text_table_contexts else "No textual context retrieved."
        
        prompt_text = (
            f"### RETRIEVED FINANCIAL CONTEXT (Text & Tables):\n{combined_text_context}\n\n"
            f"### USER QUESTION:\n{question}\n\n"
            f"Please synthesize all retrieved evidence (including any visual charts/images provided) and answer thoroughly:"
        )

        # Combine image parts and text prompt
        contents: list[Any] = []
        contents.extend(image_parts)
        contents.append(prompt_text)

        response = self.genai_client.models.generate_content(
            model=self.settings.chat_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )

        return {
            "answer": response.text,
            "contexts": contexts,
            "retrieved_images": retrieved_images,
            "retrieved_tables": retrieved_tables,
        }
