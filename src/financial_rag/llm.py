"""Gemini LLM client for bilingual (Vietnamese / English) answer generation.

Uses the official Google GenAI SDK (``google-genai``) with fallback to ``google.generativeai``.
Instructs the model to respond in the same language as the user's question
and to cite evidence sources.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Wrapper around Google GenAI for answer generation."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key
        self.clean_model_name = model_name.removeprefix("models/")
        self._client = None
        self._legacy_model = None

        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        except ImportError:
            try:
                import google.generativeai as genai_legacy

                genai_legacy.configure(api_key=api_key)
                self._legacy_model = genai_legacy.GenerativeModel(self.clean_model_name)
            except ImportError as exc:
                raise ImportError(
                    "Either 'google-genai' or 'google-generativeai' is required. "
                    "Install with: pip install google-genai"
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        evidence_texts: list[str],
        facts_text: str = "",
    ) -> str:
        """Generate a natural-language answer from retrieved evidence.

        Parameters
        ----------
        question:
            The user's question (Vietnamese or English).
        evidence_texts:
            Retrieved text passages with source/page metadata.
        facts_text:
            Optional pre-formatted structured facts from the Knowledge Graph.

        Returns
        -------
        str
            The generated answer text.
        """
        context = "\n---\n".join(evidence_texts)
        facts_section = f"\nStructured Facts:\n{facts_text}\n" if facts_text else ""

        prompt = (
            "Bạn là trợ lý phân tích tài chính chuyên nghiệp hỗ trợ song ngữ Việt – Anh. "
            "Dựa trên các bằng chứng trích xuất bên dưới, hãy trả lời câu hỏi chính xác "
            "với trích dẫn nguồn (tên tài liệu, số trang). "
            "Nếu câu hỏi bằng tiếng Việt, trả lời bằng tiếng Việt. "
            "Nếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh. "
            "Nếu không đủ dữ liệu để trả lời, hãy nói rõ.\n"
            "Khi cần thực hiện phép tính số học (tỷ lệ tăng trưởng %, biên lợi nhuận, so sánh chênh lệch), "
            "hãy đặt biểu thức toán học vào thẻ <<<CALC: biểu_thức>>> (ví dụ: <<<CALC: (125.5 - 110.0) / 110.0 * 100>>>). "
            "Hệ thống sẽ tự động tính toán chính xác tuyệt đối.\n\n"
            "You are a professional bilingual (Vietnamese – English) financial "
            "analysis assistant. Based on the extracted evidence below, answer the "
            "question accurately with source citations (document name, page number). "
            "Answer in the same language as the question. "
            "If there is insufficient data, state that clearly.\n"
            "When performing financial arithmetic (YoY growth %, margin, variance), "
            "place the arithmetic formula in <<<CALC: expression>>> (e.g. <<<CALC: (383.3 - 383.29) / 383.29 * 100>>>). "
            "The system will automatically compute the exact result.\n\n"
            f"=== Evidence ===\n{context}\n"
            f"{facts_section}\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ===\n"
        )

        if self._client is not None:
            response = self._client.models.generate_content(
                model=self.clean_model_name,
                contents=prompt,
            )
            return response.text or ""
        elif self._legacy_model is not None:
            response = self._legacy_model.generate_content(prompt)
            return response.text or ""
        return ""

    def ocr_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        """Perform vision OCR on an image (e.g. scanned PDF page) to extract text and tables.

        Instructs Gemini Vision to extract clear structured text and convert all financial
        tables to Markdown tables.
        """
        prompt = (
            "Bạn là một hệ thống OCR tài liệu và báo cáo tài chính chuyên nghiệp cấp cao. "
            "Hãy nhận diện và chuyển đổi toàn bộ nội dung trong hình ảnh trang tài liệu này thành văn bản có cấu trúc rõ ràng.\n\n"
            "Quy tắc quan trọng:\n"
            "1. Giữ nguyên 100% độ chính xác của mọi con số, mã số chỉ tiêu, ngày tháng, tỷ lệ phần trăm và đơn vị tiền tệ (VND, USD, tỷ đồng,...).\n"
            "2. Đối với tất cả bảng biểu tài chính (Bảng cân đối kế toán, Báo cáo kết quả hoạt động kinh doanh, Báo cáo lưu chuyển tiền tệ, Thuyết minh báo cáo tài chính), "
            "BẮT BUỘC định dạng thành Markdown Table chuẩn (| Cột 1 | Cột 2 | ... |).\n"
            "3. Giữ nguyên tiêu đề phân mục (Heading 1, 2, 3), các đoạn văn bản giải trình.\n"
            "4. Không thêm lời chào, nhận xét hay giải thích của AI, chỉ trả về nội dung văn bản và bảng biểu đã trích xuất."
        )

        try:
            if self._client is not None:
                from google.genai import types

                part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                response = self._client.models.generate_content(
                    model=self.clean_model_name,
                    contents=[part, prompt],
                )
                return (response.text or "").strip()
            elif self._legacy_model is not None:
                import io

                from PIL import Image

                image = Image.open(io.BytesIO(image_bytes))
                response = self._legacy_model.generate_content([image, prompt])
                return (response.text or "").strip()
        except Exception as exc:
            logger.warning("Gemini Vision OCR failed: %s", exc)
        return ""

    def analyze_chart(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> dict | None:
        """Analyze an image to determine if it is a financial chart and extract its structured data.

        Returns a dictionary:
        {
            "is_financial_chart": bool,
            "chart_type": str,
            "title": str,
            "summary": str,
            "markdown_table": str,
            "data_points": [
                {
                    "metric": str,
                    "value": float,
                    "raw_value": str,
                    "unit": str | None,
                    "period": int | None
                }
            ]
        }
        or {"is_financial_chart": False} if not a financial chart.
        """
        prompt = (
            "Bạn là chuyên gia phân tích biểu đồ tài chính. Hãy phân tích hình ảnh này và trả về kết quả dưới định dạng JSON duy nhất tuân theo cấu trúc sau:\n"
            "{\n"
            '  "is_financial_chart": true,\n'
            '  "chart_type": "bar_chart" | "line_chart" | "pie_chart" | "column_chart" | "area_chart" | "other",\n'
            '  "title": "Tiêu đề của biểu đồ",\n'
            '  "summary": "Mô tả ngắn gọn nội dung và xu hướng chính thể hiện trong biểu đồ",\n'
            '  "markdown_table": "| Chỉ tiêu | Năm/Kỳ | Giá trị | Đơn vị |\\n|---|---|---|---|\\n...",\n'
            '  "data_points": [\n'
            '    {\n'
            '      "metric": "Tên chỉ tiêu tài chính (ví dụ: Doanh thu thuần, Lợi nhuận sau thuế, Tổng tài sản)",\n'
            '      "value": 1250.5,\n'
            '      "raw_value": "1.250,5 tỷ",\n'
            '      "unit": "tỷ VND",\n'
            '      "period": 2023\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Quy tắc quan trọng:\n"
            "1. Nếu hình ảnh KHÔNG phải là biểu đồ tài chính (ví dụ: ảnh chân dung lãnh đạo, logo công ty, ảnh phong cảnh, chữ ký, con dấu đỏ, biểu tượng trang trí, sơ đồ tổ chức không có số liệu), trả về: {\"is_financial_chart\": false}\n"
            "2. Nếu LÀ biểu đồ tài chính, trích xuất tất cả các số liệu cụ thể hiển thị trên các cột/đường/lát cắt vào `data_points` và tạo `markdown_table` tương ứng.\n"
            "3. Chỉ trả về chuỗi JSON hợp lệ, không bọc trong lời dẫn hay giải thích thêm."
        )

        try:
            raw_text = ""
            if self._client is not None:
                from google.genai import types

                part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
                response = self._client.models.generate_content(
                    model=self.clean_model_name,
                    contents=[part, prompt],
                    config=config,
                )
                raw_text = response.text or ""
            elif self._legacy_model is not None:
                import io

                from PIL import Image

                image = Image.open(io.BytesIO(image_bytes))
                response = self._legacy_model.generate_content(
                    [image, prompt],
                    generation_config={"temperature": 0.1},
                )
                raw_text = response.text or ""

            return self._parse_json(raw_text)
        except Exception as exc:
            logger.warning("Gemini Vision chart analysis failed: %s", exc)
            return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        import json
        import re

        clean = text.strip()
        if not clean:
            return None
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
        try:
            return json.loads(clean)
        except Exception:
            match = re.search(r"(\{.*\})", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception as exc:
                    logger.debug("Regex json fallback failed: %s", exc)
        return None

