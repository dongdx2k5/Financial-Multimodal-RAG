"""Unified LLM Client supporting Google Gemini and OpenAI-compatible LLaMA API providers.

Supports:
- Text generation & Chat completions
- Multimodal reasoning with Vision (Gemini Vision / Llama 3.2 Vision)
- Text embeddings (Gemini embedding-001 or OpenAI-compatible embeddings)
"""

import base64
import time
from typing import Any

from google import genai
from google.genai import types
import openai

from financial_rag.config.settings import Settings


class UnifiedLLMClient:
    """Provider-agnostic LLM client for Financial RAG."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_llama = settings.is_llama

        if self.is_llama:
            api_key = settings.active_api_key
            if not api_key:
                raise ValueError(
                    "LLAMA_API_KEY (hoặc OPENAI_API_KEY) chưa được cấu hình! "
                    "Vui lòng điền LLAMA_API_KEY vào file '.env' khi sử dụng LLM_PROVIDER=llama."
                )
            self.openai_client = openai.OpenAI(
                base_url=settings.llama_base_url,
                api_key=api_key,
            )
            # Gemini client optional for fallback embeddings
            self.genai_client = None
            if settings.gemini_api_key:
                try:
                    self.genai_client = genai.Client(api_key=settings.gemini_api_key)
                except Exception:
                    self.genai_client = None
        else:
            api_key = settings.api_key
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY chưa được cấu hình! Vui lòng điền GEMINI_API_KEY vào file '.env'."
                )
            self.genai_client = genai.Client(api_key=api_key)
            self.openai_client = None

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text from a prompt."""
        if self.is_llama and self.openai_client:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.openai_client.chat.completions.create(
                model=self.settings.llama_chat_model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        assert self.genai_client is not None
        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=temperature,
        )
        response = self.genai_client.models.generate_content(
            model=self.settings.chat_model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def generate_multimodal(
        self,
        prompt: str,
        images: list[dict[str, Any]],
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text with multimodal vision evidence (images + text).

        Parameters
        ----------
        prompt : str
            The formatted text prompt including context and question.
        images : list[dict[str, Any]]
            List of image items with keys: 'image_base64', 'mime_type'.
        system_prompt : str | None
            System guidelines.
        temperature : float
            Sampling temperature.
        """
        if self.is_llama and self.openai_client:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            user_contents: list[dict[str, Any]] = []
            for img in images:
                b64 = img.get("image_base64")
                mime = img.get("mime_type", "image/png")
                if b64 and isinstance(b64, str):
                    user_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
            user_contents.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_contents})

            response = self.openai_client.chat.completions.create(
                model=self.settings.llama_chat_model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        assert self.genai_client is not None
        contents: list[Any] = []
        for img in images:
            b64 = img.get("image_base64")
            mime = img.get("mime_type", "image/png")
            if b64 and isinstance(b64, str):
                try:
                    img_bytes = base64.b64decode(b64)
                    contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                except Exception:
                    pass

        contents.append(prompt)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=temperature,
        )
        response = self.genai_client.models.generate_content(
            model=self.settings.chat_model,
            contents=contents,
            config=config,
        )
        return response.text or ""

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of text strings."""
        if not texts:
            return []

        # If LLaMA is active and has a configured embedding model
        if self.is_llama and self.settings.llama_embedding_model and self.openai_client:
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.openai_client.embeddings.create(
                    model=self.settings.llama_embedding_model,
                    input=batch,
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings

        # Gemini embeddings (default or fallback)
        if self.genai_client:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                success = False
                for attempt in range(7):
                    try:
                        resp = self.genai_client.models.embed_content(
                            model=self.settings.embedding_model,
                            contents=batch,
                        )
                        all_embeddings.extend([item.values for item in resp.embeddings])
                        success = True
                        break
                    except Exception as e:
                        err_msg = str(e)
                        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                            time.sleep(6 * (attempt + 1))
                        else:
                            raise e
                if not success:
                    raise RuntimeError(f"Failed to embed batch starting at index {i} after 7 retries.")
                time.sleep(0.3)
            return all_embeddings

        raise ValueError(
            "Không thể tạo embedding: không có Gemini API key và cũng không cấu hình LLAMA_EMBEDDING_MODEL."
        )
