"""LLM backend interface supporting Qwen2.5-3B-Instruct, Mock generator, and OpenAI."""

import logging
from typing import Optional
from financial_rag.config import settings
from financial_rag.llm.prompt import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


class LLMBackend:
    """Interface for text generation with Qwen2.5-3B-Instruct on Colab GPU or mock mode."""

    def __init__(
        self,
        backend_type: Optional[str] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.backend_type = backend_type or settings.llm_backend
        self.model_id = model_id or settings.llm_model_id
        self.device = device or settings.llm_device

        self.tokenizer = None
        self.model = None

    def _init_qwen_local(self):
        if self.model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"Loading local model {self.model_id} on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                torch_dtype = torch.float16 if settings.llm_fp16 and torch.cuda.is_available() else torch.float32
                device_target = self.device if torch.cuda.is_available() else "cpu"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto" if device_target == "cuda" else None,
                    trust_remote_code=True,
                )
                if device_target == "cpu":
                    self.model.to("cpu")
                logger.info(f"Model {self.model_id} loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load local Qwen model: {e}. Falling back to mock generator.")
                self.backend_type = "mock"

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate text strictly with do_sample=False (temperature 0)."""
        if self.backend_type == "qwen_local":
            self._init_qwen_local()
            if self.model is not None and self.tokenizer is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Temperature 0.0
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated_ids = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response.strip()

        elif self.backend_type == "openai" and settings.openai_api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=settings.openai_api_key)
                res = client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                return (res.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(f"OpenAI error: {e}. Falling back to mock generator.")

        # Default Mock / Rule-based generator
        return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """Deterministic mock response formatter for offline testing."""
        # Check if prompt contains Calculation/Factual Grounding
        if "Calculation/Factual Grounding:" in prompt:
            lines = prompt.splitlines()
            grounding_lines = []
            citations_lines = []
            capturing_g = False
            capturing_c = False
            for line in lines:
                if "Calculation/Factual Grounding:" in line:
                    capturing_g = True
                    capturing_c = False
                    continue
                elif "Citations:" in line:
                    capturing_g = False
                    capturing_c = True
                    continue
                elif "Instructions:" in line:
                    capturing_g = False
                    capturing_c = False
                    continue

                if capturing_g and line.strip():
                    grounding_lines.append(line.strip())
                elif capturing_c and line.strip():
                    citations_lines.append(line.strip())

            content = " ".join(grounding_lines)
            citations = " ".join(citations_lines)
            return f"{content}\n\nNguồn trích dẫn: {citations}"

        elif "Retrieved Excerpts:" in prompt:
            # Extract first relevant sentence or citation
            for line in prompt.splitlines():
                if "[" in line and "Trang" in line:
                    return f"Theo tài liệu công bố, {line.strip()}"
            return "Theo báo cáo tài chính, thông tin chi tiết đã được phản ánh trong phần thuyết minh đính kèm."

        return "Thông tin đã được tra cứu và xác thực từ cơ sở dữ liệu báo cáo tài chính."
