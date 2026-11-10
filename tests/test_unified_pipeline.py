"""
Bộ Kiểm thử Tự động Toàn diện (Unit & Integration Tests) cho Unified Financial Multimodal RAG
=============================================================================================
Hỗ trợ chạy bằng cả `pytest test_unified_pipeline.py` lẫn `python3 test_unified_pipeline.py`.
Kiểm chứng toàn bộ các tính năng hợp nhất từ 2 nhánh và 8 lỗi trong `REVIEW_20261020.md`:
1. [Bug #1 High] Phát hiện file PDF sai doanh nghiệp/kỳ báo cáo (`NVDA_10K_2024_sample.pdf` chứa Analog Devices 2009).
2. [Bug #2 High] Xử lý trang PDF scan không có text layer qua `SharedQwenVLBackbone` + `validate_extraction` chuẩn.
3. [Bug #3 High] Kết nối khép kín End-to-End Router + HybridRetriever + Deterministic Calculator + NumericalVerifier.
4. [Bug #4 Medium] `UnifiedSQLiteStore.query_fact()` fallback có thứ tự `ORDER BY period DESC` kèm nhãn kỳ thực tế.
5. [Bug #5 Medium] `FactExtractor.clean_number()` bảo toàn số âm `(1.234.567)` và `-1234.5` cũng như số `0`.
6. [Bug #6 Medium] `NumericalVerifier.extract_numbers()` kiểm chứng cả các con số nhỏ `<= 10` (biên lợi nhuận, % tăng trưởng).
7. [VAS Quality Gate] Kiểm tra chéo đẳng thức kế toán VAS (`270=100+200`, `440=300+400`, `20=10-11`, `50=20+30+40`)
   và vòng lặp `Self-Correction` tự động đọc lại bảng khi lệch số liệu.
"""

from __future__ import annotations

import unittest
from unified_financial_rag_colab_t4 import (
    FactExtractor,
    FinancialFact,
    NumericalVerifier,
    SharedQwenVLBackbone,
    SmartFinancialPDFParser,
    UnifiedFinancialRAGPipeline,
)


def test_bug1_detect_mislabeled_nvda_analog_devices_pdf():
    """Kiểm tra Bug #1: Chặn ngay file gắn nhãn NVDA FY2024 nhưng nội dung là Analog Devices 2009."""
    parser = SmartFinancialPDFParser()
    fake_nvda_pages = [
        {
            "page": 1,
            "text": "Analog Devices, Inc. 2009 Annual Report. Consolidated Statements of Income for fiscal 2009.",
            "tables": [],
        }
    ]
    raised = False
    error_msg = ""
    try:
        parser.extract_from_pages_data(
            pages_data=fake_nvda_pages,
            company="NVDA",
            period="FY2024",
            strict_identity_check=True,
        )
    except ValueError as exc:
        raised = True
        error_msg = str(exc)

    assert raised is True, "Phải ném ValueError khi nội dung PDF không khớp với nhãn công ty"
    assert "NVDA" in error_msg
    assert "ADI" in error_msg


def test_bug2_scanned_pdf_vlm_ocr_and_self_correction():
    """
    Kiểm tra Bug #2 & VAS Self-Correction:
    - Khi trang PDF scan không có text layer, nếu VLM trả về rỗng thì `validate_extraction` phải báo `is_valid=False`.
    - Khi VLM lần 1 đọc nhầm Mã 270 (Tổng tài sản = 350.000 thay vì 300.000 = 100.000 + 200.000),
      `VASAccountingQualityGate` phát hiện lỗi và tự động gọi lại VLM với `correction_hint` để sửa thành 300.000.
    """
    empty_parser = SmartFinancialPDFParser(
        vlm_backbone=SharedQwenVLBackbone(mock_vlm_callable=lambda prompt, img: "[]")
    )
    empty_res = empty_parser.extract_from_pages_data(
        pages_data=[{"page": 1, "text": "", "tables": [], "image": "blank_scan.png"}],
        company="FPT",
        period="FY2024",
    )
    assert empty_res["validation"]["is_valid"] is False
    assert empty_res["validation"]["num_facts"] == 0

    call_counter = {"count": 0}

    def mock_vlm_with_self_correction(prompt: str, img):
        call_counter["count"] += 1
        if "LƯU Ý KIỂM TOÁN CẦN ĐỐI CHIẾU LẠI" not in prompt:
            return """[
                ["Tài sản ngắn hạn", "100", "V.01", "100.000", "80.000"],
                ["Tài sản dài hạn", "200", "V.02", "200.000", "150.000"],
                ["Tổng cộng tài sản", "270", "", "350.000", "230.000"],
                ["Nợ phải trả", "300", "V.10", "120.000", "90.000"],
                ["Vốn chủ sở hữu", "400", "V.15", "180.000", "140.000"],
                ["Tổng cộng nguồn vốn", "440", "", "300.000", "230.000"]
            ]"""
        return """[
            ["Tài sản ngắn hạn", "100", "V.01", "100.000", "80.000"],
            ["Tài sản dài hạn", "200", "V.02", "200.000", "150.000"],
            ["Tổng cộng tài sản", "270", "", "300.000", "230.000"],
            ["Nợ phải trả", "300", "V.10", "120.000", "90.000"],
            ["Vốn chủ sở hữu", "400", "V.15", "180.000", "140.000"],
            ["Tổng cộng nguồn vốn", "440", "", "300.000", "230.000"]
        ]"""

    pipeline = UnifiedFinancialRAGPipeline(
        vlm_backbone=SharedQwenVLBackbone(mock_vlm_callable=mock_vlm_with_self_correction)
    )
    res = pipeline.ingest_pages(
        pages_data=[{"page": 5, "text": "", "tables": [], "image": "fpt_bs_scan.png"}],
        company="FPT",
        period="FY2024",
    )
    assert call_counter["count"] == 2
    assert res["quality_report"].self_correction_triggered is True
    assert res["quality_report"].overall_status == "pass"
    assert res["validation"]["is_valid"] is True


def test_bug4_sqlite_period_fallback_ordering_and_notice():
    """
    Kiểm tra Bug #4: Khi kỳ được hỏi (FY2025) chưa có trong DB, `query_fact` phải trả về
    kỳ mới nhất (`FY2024` thay vì `FY2022`) và câu trả lời phải thông báo rõ kỳ thực tế.
    """
    pipeline = UnifiedFinancialRAGPipeline()
    pipeline.store.upsert_facts([
        FinancialFact(company="FPT", period="FY2022", metric_code="net_revenue", metric_name="Doanh thu thuần", value=44000.0, vas_code="10", statement_type="income_statement"),
        FinancialFact(company="FPT", period="FY2024", metric_code="net_revenue", metric_name="Doanh thu thuần", value=62849.0, vas_code="10", statement_type="income_statement"),
        FinancialFact(company="FPT", period="FY2023", metric_code="net_revenue", metric_name="Doanh thu thuần", value=52618.0, vas_code="10", statement_type="income_statement"),
    ])

    ans = pipeline.answer_question("Doanh thu thuần của FPT là bao nhiêu?", company="FPT", period="FY2025")
    assert ans["fact"]["actual_period"] == "FY2024"
    assert ans["fact"]["is_fallback_period"] is True
    assert "FY2024" in ans["answer"]
    assert "62,849" in ans["answer"]


def test_bug5_negative_numbers_and_zero_preserved():
    """
    Kiểm tra Bug #5: Các khoản lỗ hoặc lưu chuyển tiền thuần âm dạng `(1.250.500)` hoặc `-450.25`
    và số `0` phải được giữ nguyên.
    """
    extractor = FactExtractor()
    assert extractor._clean_number("(1.250.500)") == -1250500.0
    assert extractor._clean_number("-450.25") == -450.25
    assert extractor._clean_number("0") == 0.0

    rows = [
        ["Lưu chuyển tiền thuần từ hoạt động đầu tư", "30", "", "(5.420.000)", "(3.100.000)"],
    ]
    facts = extractor.extract_from_table_rows(rows, company="FPT", period="FY2024", statement_type="cash_flow")
    assert len(facts) == 1
    assert facts[0].value == -5420000.0
    assert facts[0].metric_code == "investing_cash_flow"


def test_bug6_numerical_verifier_checks_small_percentages_and_ratios():
    """
    Kiểm tra Bug #6: `NumericalVerifier` phải bắt được các số nhỏ `<= 10` (ví dụ biên lợi nhuận 8.5%
    bị LLM bịa thành 9.8%).
    """
    verifier = NumericalVerifier()
    good_check = verifier.verify(
        "Tăng trưởng doanh thu năm 2024 đạt 6.45% và hệ số thanh toán là 1.82 lần.",
        reference_numbers=[6.45, 1.82],
    )
    assert good_check["is_verified"] is True
    assert 6.45 in good_check["verified_numbers"]
    assert 1.82 in good_check["verified_numbers"]

    hallucinated_check = verifier.verify(
        "Biên lợi nhuận ròng năm 2024 đạt 9.75%.",
        reference_numbers=[6.45, 1.82],
    )
    assert hallucinated_check["is_verified"] is False
    assert 9.75 in hallucinated_check["unverified_numbers"]


def test_end_to_end_deterministic_yoy_ratio_and_reconciliation():
    """
    Kiểm tra Bug #3: Toàn bộ luồng tính toán YoY, ROE/Biên lợi nhuận gộp và Kiểm toán BCTC
    hoạt động khép kín với độ chính xác Decimal.
    """
    pipeline = UnifiedFinancialRAGPipeline()
    pages_2023 = [
        {
            "page": 2,
            "text": "Báo cáo kết quả hoạt động kinh doanh năm 2023",
            "tables": [
                [
                    ["Doanh thu thuần về bán hàng và cung cấp dịch vụ", "10", "VI.1", "52.618", "44.000"],
                    ["Giá vốn hàng bán", "11", "VI.2", "32.000", "27.000"],
                    ["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "20", "", "20.618", "17.000"],
                ]
            ],
        }
    ]
    pages_2024 = [
        {
            "page": 2,
            "text": "Báo cáo kết quả hoạt động kinh doanh năm 2024",
            "tables": [
                [
                    ["Doanh thu thuần về bán hàng và cung cấp dịch vụ", "10", "VI.1", "62.849", "52.618"],
                    ["Giá vốn hàng bán", "11", "VI.2", "38.249", "32.000"],
                    ["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "20", "", "24.600", "20.618"],
                ]
            ],
        }
    ]
    pipeline.ingest_pages(pages_2023, company="FPT", period="FY2023")
    pipeline.ingest_pages(pages_2024, company="FPT", period="FY2024")

    # 1. Hỏi tăng trưởng YoY
    yoy_out = pipeline.answer_question(
        "Tăng trưởng doanh thu thuần của FPT năm 2024 so với 2023?",
        company="FPT",
        period="FY2024",
        comparison_period="FY2023",
    )
    assert yoy_out["execution_path"] == "deterministic_calculator"
    assert yoy_out["calculation"]["pct_change"] == 19.44
    assert yoy_out["verification"]["is_verified"] is True

    # 2. Hỏi Biên lợi nhuận gộp
    margin_out = pipeline.answer_question(
        "Tính biên lợi nhuận gộp của FPT kỳ FY2024",
        company="FPT",
        period="FY2024",
    )
    assert margin_out["execution_path"] == "deterministic_calculator"
    assert margin_out["calculation"]["value"] == 39.14
    assert margin_out["verification"]["is_verified"] is True

    # 3. Hỏi báo cáo kiểm toán cân đối BCTC
    audit_out = pipeline.answer_question(
        "Kiểm toán đẳng thức kế toán BCTC FPT",
        company="FPT",
        period="FY2024",
    )
    assert audit_out["execution_path"] == "deterministic_reconciliation"
    assert "ĐẠT CHUẨN" in audit_out["answer"]




def test_upgrade1_unit_scale_normalizer():
    """Kiểm tra Nâng cấp #1: Tự động quy đổi 'Đơn vị tính: Triệu đồng' và 'Tỷ đồng' về VND chuẩn."""
    from unified_financial_rag_colab_t4 import UnitScaleNormalizer

    unit, mult = UnitScaleNormalizer.detect_unit_and_multiplier("Bảng cân đối kế toán - Đơn vị tính: Triệu đồng")
    assert unit == "VND"
    assert mult == 1_000_000.0

    raw_facts = [
        FinancialFact(company="FPT", period="FY2024", metric_code="net_revenue", metric_name="Doanh thu thuần", value=62849.0, unit="VND")
    ]
    scaled = UnitScaleNormalizer.normalize_facts(raw_facts, "Đơn vị tính: Tỷ đồng")
    assert scaled[0].value == 62_849_000_000_000.0


def test_upgrade2_report_scope_detector():
    """Kiểm tra Nâng cấp #2: Phân biệt BCTC Hợp nhất (consolidated) và BCTC Riêng (separate)."""
    from unified_financial_rag_colab_t4 import ReportScopeDetector

    assert ReportScopeDetector.detect_scope("BÁO CÁO TÀI CHÍNH HỢP NHẤT NĂM 2024") == "consolidated"
    assert ReportScopeDetector.detect_scope("BÁO CÁO TÀI CHÍNH RIÊNG CÔNG TY MẸ NĂM 2024") == "separate"


def test_upgrade3_vlm_ocr_sqlite_cache():
    """Kiểm tra Nâng cấp #3: Lưu cache SHA-256 cho kết quả đọc bảng của Qwen2.5-VL trên Colab T4."""
    from unified_financial_rag_colab_t4 import PageOCRCacheManager, UnifiedSQLiteStore

    store = UnifiedSQLiteStore(":memory:")
    cache = PageOCRCacheManager(store.conn)
    h = cache.compute_hash("fpt_page_5_image_bytes")
    assert cache.get_cached_rows(h) is None

    sample_rows = [["Tổng cộng tài sản", "270", "", "300.000", "230.000"]]
    cache.save_cached_rows(h, sample_rows, statement_hint="balance_sheet")
    loaded = cache.get_cached_rows(h)
    assert loaded == sample_rows




def test_upgrade4_default_seed_and_real_pdf_file_ingestion():
    """
    Kiểm tra khắc phục lỗi không nạp được tài liệu trên Gradio:
    1. Khi khởi tạo `UnifiedFinancialRAGPipeline()`, dữ liệu mẫu FPT FY2023 & FY2024 được nạp sẵn
       để bấm nút hỏi ngay trên Gradio luôn trả về số liệu chuẩn xác.
    2. Tạo một file `.pdf` thực tế trên đĩa và gọi `ingest_pdf_file()` để kiểm chứng luồng Upload PDF.
    """
    import tempfile
    from pathlib import Path

    pipeline = UnifiedFinancialRAGPipeline(seed_sample_data_if_empty=True)
    summary = pipeline.get_database_summary()
    assert summary["total_facts"] >= 15

    # Hỏi ngay lập tức mà không cần nạp thủ công
    immediate_ans = pipeline.answer_question(
        "Tổng cộng tài sản của FPT năm 2024 là bao nhiêu?",
        company="FPT",
        period="FY2024",
    )
    assert "72,000" in immediate_ans["answer"]
    assert immediate_ans["verification"]["is_verified"] is True

    # Kiểm thử tạo file PDF vật lý và nạp qua `ingest_pdf_file`
    from pypdf import PdfWriter
    import io
    try:
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(50, 780, "CONG TY CO PHAN SUA VIET NAM (VNM) - BAO CAO TAI CHINH HOP NHAT 2024")
        c.drawString(50, 750, "Bang can doi ke toan hop nhat")
        c.drawString(50, 720, "Tai san ngan han   100   V.01   35.000   31.000")
        c.drawString(50, 700, "Tai san dai han    200   V.02   20.000   19.000")
        c.drawString(50, 680, "Tong cong tai san  270          55.000   50.000")
        c.save()
        pdf_bytes = buf.getvalue()
    except Exception:
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "VNM_BCTC_2024.pdf"
        pdf_path.write_bytes(pdf_bytes)
        ingest_res = pipeline.ingest_pdf_file(pdf_path)
        assert ingest_res["company"] == "VNM"
        assert ingest_res["period"] == "FY2024"
        assert ingest_res["pages_read"] == 1


class TestUnifiedFinancialPipelineSuite(unittest.TestCase):
    def test_07_unit_scale_normalizer(self):
        test_upgrade1_unit_scale_normalizer()

    def test_08_report_scope_detector(self):
        test_upgrade2_report_scope_detector()

    def test_09_vlm_ocr_sqlite_cache(self):
        test_upgrade3_vlm_ocr_sqlite_cache()

    def test_10_default_seed_and_real_pdf_file_ingestion(self):
        test_upgrade4_default_seed_and_real_pdf_file_ingestion()

    def test_01_mislabeled_pdf(self):
        test_bug1_detect_mislabeled_nvda_analog_devices_pdf()

    def test_02_scanned_pdf_and_self_correction(self):
        test_bug2_scanned_pdf_vlm_ocr_and_self_correction()

    def test_03_sqlite_period_fallback(self):
        test_bug4_sqlite_period_fallback_ordering_and_notice()

    def test_04_negative_and_zero_numbers(self):
        test_bug5_negative_numbers_and_zero_preserved()

    def test_05_numerical_verifier_small_numbers(self):
        test_bug6_numerical_verifier_checks_small_percentages_and_ratios()

    def test_06_end_to_end_deterministic_qa(self):
        test_end_to_end_deterministic_yoy_ratio_and_reconciliation()


if __name__ == "__main__":
    unittest.main(verbosity=2)
