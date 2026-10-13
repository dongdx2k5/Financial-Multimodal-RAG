"""Phase 2 tests – Multimodal Vision OCR for Scanned PDFs (2.1) & Chart Understanding (2.2).

These tests run fully offline using mocks – no external LLM / API key required.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import PIL.Image
import pymupdf
import pytest

from financial_rag.config import Settings
from financial_rag.ingestion import DocumentLoader
from financial_rag.llm import GeminiLLM
from financial_rag.service import FinancialRAGService

# =====================================================================
# Helpers to generate synthetic PDFs for testing
# =====================================================================

def _create_scanned_pdf(path: Path, min_text: str = "") -> Path:
    """Create a PDF with little or no selectable text (< 50 chars)."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    if min_text:
        page.insert_text(pymupdf.Point(50, 50), min_text)
    # Render an image onto the page to simulate a scan
    img = PIL.Image.new("RGB", (500, 700), color=(250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 50, 545, 750), stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


def _create_pdf_with_chart(path: Path, text: str = "Báo cáo thường niên 2024") -> Path:
    """Create a PDF containing normal text plus an embedded chart image."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Insert > 50 characters of text so page itself is not considered scanned
    long_text = f"{text}\n" + ("Nội dung giải trình kết quả sản xuất kinh doanh chi tiết trong năm. " * 3)
    page.insert_text(pymupdf.Point(50, 50), long_text)

    # Insert an image >= 100x100 and >= 2000 bytes
    img = PIL.Image.new("RGB", (300, 200), color=(80, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 150, 350, 350), stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


def _create_pdf_with_small_icon(path: Path) -> Path:
    """Create a PDF containing a tiny icon (<100x100) that should be ignored."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(pymupdf.Point(50, 50), "Trang có chứa logo biểu tượng nhỏ không phải đồ thị." * 2)

    # Insert a tiny 20x20 icon
    img = PIL.Image.new("RGB", (20, 20), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 80, 70, 100), stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


# =====================================================================
# Bước 2.1 – Gemini Vision OCR cho PDF Scan
# =====================================================================

class TestVisionOCRScannedPDF:
    def test_scanned_page_detection_flag(self, tmp_path: Path):
        pdf_path = _create_scanned_pdf(tmp_path / "scanned_doc.pdf", min_text="Scan")
        loader = DocumentLoader()
        doc = loader.load(pdf_path, ticker="HPG", year=2024)

        assert len(doc.pages) == 1
        assert doc.pages[0].is_scanned is True

    def test_normal_text_page_not_flagged_scanned(self, tmp_path: Path):
        pdf_path = tmp_path / "normal_text.pdf"
        doc_pdf = pymupdf.open()
        p = doc_pdf.new_page()
        p.insert_text(
            pymupdf.Point(50, 50),
            "Doanh thu thuần về bán hàng và cung cấp dịch vụ năm 2024 đạt 125,5 tỷ VND.",
        )
        doc_pdf.save(str(pdf_path))
        doc_pdf.close()

        loader = DocumentLoader()
        doc = loader.load(pdf_path, ticker="HPG", year=2024)

        assert len(doc.pages) == 1
        assert doc.pages[0].is_scanned is False

    def test_vision_ocr_invoked_on_scanned_page(self, tmp_path: Path):
        pdf_path = _create_scanned_pdf(tmp_path / "scan_table.pdf")
        mock_vision = MagicMock()
        mock_vision.ocr_image.return_value = (
            "BÁO CÁO KẾT QUẢ KINH DOANH\n"
            "| Chỉ tiêu | Năm 2024 |\n"
            "|---|---|\n"
            "| Doanh thu thuần | 150.0 tỷ VND |\n"
            "| Lợi nhuận sau thuế | 25.0 tỷ VND |\n"
        )

        loader = DocumentLoader(vision_llm=mock_vision)
        doc = loader.load(pdf_path, ticker="HPG", year=2024)

        assert mock_vision.ocr_image.called
        assert doc.pages[0].is_scanned is True
        assert "150.0 tỷ VND" in doc.pages[0].text
        assert "Lợi nhuận sau thuế" in doc.pages[0].text

    def test_vision_ocr_offline_graceful_fallback(self, tmp_path: Path):
        pdf_path = _create_scanned_pdf(tmp_path / "scan_offline.pdf")
        # No vision_llm configured
        loader = DocumentLoader(vision_llm=None)
        doc = loader.load(pdf_path, ticker="HPG", year=2024)

        assert len(doc.pages) == 1
        assert doc.pages[0].is_scanned is True

    def test_gemini_llm_ocr_image_with_mocked_client(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "| Doanh thu thuần | 200 tỷ |"
            mock_client.models.generate_content.return_value = mock_response

            llm = GeminiLLM(api_key="test-key", model_name="gemini-3.8-flash")
            ocr_result = llm.ocr_image(b"fake_png_bytes", mime_type="image/png")

            assert "| Doanh thu thuần | 200 tỷ |" in ocr_result
            assert mock_client.models.generate_content.called


# =====================================================================
# Bước 2.2 – Phân tích biểu đồ (Chart Understanding)
# =====================================================================

class TestChartUnderstanding:
    def test_small_icon_is_skipped(self, tmp_path: Path):
        pdf_path = _create_pdf_with_small_icon(tmp_path / "with_icon.pdf")
        mock_vision = MagicMock()

        loader = DocumentLoader(vision_llm=mock_vision)
        doc = loader.load(pdf_path, ticker="DEMO", year=2024)

        # analyze_chart should NOT be called on tiny icons
        assert not mock_vision.analyze_chart.called
        assert len(doc.charts) == 0

    def test_non_financial_image_filtered_out(self, tmp_path: Path):
        pdf_path = _create_pdf_with_chart(tmp_path / "with_photo.pdf")
        mock_vision = MagicMock()
        mock_vision.analyze_chart.return_value = {
            "is_financial_chart": False,
        }

        loader = DocumentLoader(vision_llm=mock_vision)
        doc = loader.load(pdf_path, ticker="DEMO", year=2024)

        assert mock_vision.analyze_chart.called
        assert len(doc.charts) == 0

    def test_financial_chart_parsed_and_appended(self, tmp_path: Path):
        pdf_path = _create_pdf_with_chart(tmp_path / "with_chart.pdf")
        mock_vision = MagicMock()
        mock_vision.analyze_chart.return_value = {
            "is_financial_chart": True,
            "chart_type": "bar_chart",
            "title": "Biểu đồ Tăng trưởng Doanh thu 2022 - 2024",
            "summary": "Doanh thu thuần tăng trưởng liên tục qua 3 năm liên tiếp.",
            "markdown_table": (
                "| Chỉ tiêu | Năm | Giá trị | Đơn vị |\n"
                "|---|---|---|---|\n"
                "| Doanh thu thuần | 2022 | 800.0 | tỷ VND |\n"
                "| Doanh thu thuần | 2023 | 950.0 | tỷ VND |\n"
                "| Doanh thu thuần | 2024 | 1200.0 | tỷ VND |\n"
            ),
            "data_points": [
                {
                    "metric": "Doanh thu thuần",
                    "value": 800.0,
                    "raw_value": "800.0 tỷ",
                    "unit": "tỷ VND",
                    "period": 2022,
                },
                {
                    "metric": "Doanh thu thuần",
                    "value": 950.0,
                    "raw_value": "950.0 tỷ",
                    "unit": "tỷ VND",
                    "period": 2023,
                },
                {
                    "metric": "Doanh thu thuần",
                    "value": 1200.0,
                    "raw_value": "1200.0 tỷ",
                    "unit": "tỷ VND",
                    "period": 2024,
                },
            ],
        }

        loader = DocumentLoader(vision_llm=mock_vision)
        doc = loader.load(pdf_path, ticker="HPG", year=2024)

        assert len(doc.charts) == 1
        chart = doc.charts[0]
        assert chart.title == "Biểu đồ Tăng trưởng Doanh thu 2022 - 2024"
        assert len(chart.data_points) == 3
        assert chart.data_points[2].value == 1200.0
        assert chart.data_points[2].period == 2024

        # Verify chart content is appended to page text for retrieval chunking
        page_text = doc.pages[0].text
        assert "BIỂU ĐỒ TÀI CHÍNH" in page_text
        assert "1200.0" in page_text

    def test_gemini_llm_analyze_chart_with_mocked_client(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = (
                '{\n'
                '  "is_financial_chart": true,\n'
                '  "chart_type": "pie_chart",\n'
                '  "title": "Cơ cấu tài sản",\n'
                '  "summary": "Tài sản ngắn hạn chiếm tỷ trọng lớn.",\n'
                '  "markdown_table": "| Chỉ tiêu | Giá trị |\\n|---|---|\\n| Tài sản ngắn hạn | 60% |",\n'
                '  "data_points": [\n'
                '    {"metric": "Tài sản ngắn hạn", "value": 60.0, "raw_value": "60%", "unit": "%", "period": 2024}\n'
                '  ]\n'
                '}'
            )
            mock_client.models.generate_content.return_value = mock_response

            llm = GeminiLLM(api_key="test-key", model_name="gemini-3.8-flash")
            result = llm.analyze_chart(b"fake_chart_bytes", mime_type="image/png")

            assert result is not None
            assert result.get("is_financial_chart") is True
            assert result.get("title") == "Cơ cấu tài sản"
            assert len(result.get("data_points")) == 1


# =====================================================================
# End-to-End Ingestion, Database & Knowledge Graph Integration
# =====================================================================

class TestMultimodalServiceIntegration:
    @pytest.fixture()
    def service(self, tmp_path: Path) -> FinancialRAGService:
        settings = Settings(
            data_dir=tmp_path / "data",
            chunk_size=500,
            chunk_overlap=50,
        )
        svc = FinancialRAGService(settings)
        return svc

    def test_ingest_pdf_with_chart_stores_facts_and_graph(self, tmp_path: Path, service: FinancialRAGService):
        pdf_path = _create_pdf_with_chart(tmp_path / "test_report.pdf")

        # Mock the vision client on the service loader
        mock_vision = MagicMock()
        mock_vision.analyze_chart.return_value = {
            "is_financial_chart": True,
            "chart_type": "bar_chart",
            "title": "Doanh thu thuần giai đoạn 2023 - 2024",
            "summary": "Doanh thu năm 2024 đạt mốc kỷ lục.",
            "markdown_table": (
                "| Chỉ tiêu | Năm 2023 | Năm 2024 |\n"
                "|---|---|---|\n"
                "| Doanh thu thuần | 500.0 tỷ VND | 750.0 tỷ VND |\n"
            ),
            "data_points": [
                {
                    "metric": "Doanh thu thuần",
                    "value": 500.0,
                    "raw_value": "500.0 tỷ",
                    "unit": "tỷ VND",
                    "period": 2023,
                },
                {
                    "metric": "Doanh thu thuần",
                    "value": 750.0,
                    "raw_value": "750.0 tỷ",
                    "unit": "tỷ VND",
                    "period": 2024,
                },
            ],
        }
        service.loader.vision_llm = mock_vision

        result = service.ingest(pdf_path, ticker="HPG", year=2024)

        assert result["charts"] == 1
        assert result["facts"] >= 2

        # 1. Verify facts in SQLite Database
        chart_facts = service.get_facts(ticker="HPG", is_chart=True)
        assert len(chart_facts) == 2
        periods = sorted([f.period for f in chart_facts])
        assert periods == [2023, 2024]
        assert all(f.is_chart is True for f in chart_facts)
        assert all(f.standardized_metric == "REVENUE_NET" for f in chart_facts)

        # 2. Verify facts in Knowledge Graph
        hist_facts = service.find_historical_facts("HPG", "REVENUE_NET", limit_years=3)
        assert len(hist_facts) >= 2
        assert hist_facts[-1].value == 750.0
        assert hist_facts[-1].period == 2024

        # 3. Verify retrieval finds evidence containing chart data
        response = service.ask("Theo biểu đồ, doanh thu thuần năm 2024 là bao nhiêu?", ticker="HPG")
        assert len(response.evidence) > 0
        found_in_evidence = any("750.0" in ev.text for ev in response.evidence)
        assert found_in_evidence
