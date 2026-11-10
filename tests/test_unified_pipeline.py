"""
Bộ Kiểm thử Tự động Toàn diện (12 Unit & Integration Tests) cho Unified Financial Multimodal RAG
================================================================================================
Hỗ trợ chạy bằng cả `pytest test_unified_pipeline.py` lẫn `python3 test_unified_pipeline.py`.
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unicodedata
import unittest

from unified_financial_rag_colab_t4 import (
    FactExtractor,
    FinancialFact,
    NumericalVerifier,
    PageOCRCacheManager,
    ReportScopeDetector,
    SharedQwenVLBackbone,
    SmartFinancialPDFParser,
    UnifiedFinancialRAGPipeline,
    UnifiedSQLiteStore,
    UnitScaleNormalizer,
)


def test_bug1_detect_mislabeled_nvda_analog_devices_pdf():
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

    assert raised is True
    assert "NVDA" in error_msg
    assert "ADI" in error_msg


def test_bug2_scanned_pdf_vlm_ocr_and_self_correction():
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

    yoy_out = pipeline.answer_question(
        "Tăng trưởng doanh thu thuần của FPT năm 2024 so với 2023?",
        company="FPT",
        period="FY2024",
        comparison_period="FY2023",
    )
    assert yoy_out["execution_path"] == "deterministic_calculator"
    assert yoy_out["calculation"]["pct_change"] == 19.44
    assert yoy_out["verification"]["is_verified"] is True

    margin_out = pipeline.answer_question(
        "Tính biên lợi nhuận gộp của FPT kỳ FY2024",
        company="FPT",
        period="FY2024",
    )
    assert margin_out["execution_path"] == "deterministic_calculator"
    assert margin_out["calculation"]["value"] == 39.14
    assert margin_out["verification"]["is_verified"] is True

    audit_out = pipeline.answer_question(
        "Kiểm toán đẳng thức kế toán BCTC FPT",
        company="FPT",
        period="FY2024",
    )
    assert audit_out["execution_path"] == "deterministic_reconciliation"
    assert "ĐẠT CHUẨN" in audit_out["answer"]


def test_upgrade1_unit_scale_normalizer():
    unit, mult = UnitScaleNormalizer.detect_unit_and_multiplier("Bảng cân đối kế toán - Đơn vị tính: Triệu đồng")
    assert unit == "VND"
    assert mult == 1_000_000.0

    raw_facts = [
        FinancialFact(company="FPT", period="FY2024", metric_code="net_revenue", metric_name="Doanh thu thuần", value=62849.0, unit="VND")
    ]
    scaled = UnitScaleNormalizer.normalize_facts(raw_facts, "Đơn vị tính: Tỷ đồng")
    assert scaled[0].value == 62_849_000_000_000.0


def test_upgrade2_report_scope_detector():
    assert ReportScopeDetector.detect_scope("BÁO CÁO TÀI CHÍNH HỢP NHẤT NĂM 2024") == "consolidated"
    assert ReportScopeDetector.detect_scope("BÁO CÁO TÀI CHÍNH RIÊNG CÔNG TY MẸ NĂM 2024") == "separate"


def test_upgrade3_vlm_ocr_sqlite_cache():
    store = UnifiedSQLiteStore(":memory:")
    cache = PageOCRCacheManager(store.conn)
    h = cache.compute_hash("fpt_page_5_image_bytes")
    assert cache.get_cached_rows(h) is None

    sample_rows = [["Tổng cộng tài sản", "270", "", "300.000", "230.000"]]
    cache.save_cached_rows(h, sample_rows, statement_hint="balance_sheet")
    loaded = cache.get_cached_rows(h)
    assert loaded == sample_rows


def test_upgrade4_default_seed_and_real_pdf_file_ingestion():
    pipeline = UnifiedFinancialRAGPipeline(seed_sample_data_if_empty=True)
    summary = pipeline.get_database_summary()
    assert summary["total_facts"] >= 15

    immediate_ans = pipeline.answer_question(
        "Tổng cộng tài sản của FPT năm 2024 là bao nhiêu?",
        company="FPT",
        period="FY2024",
    )
    assert "72,000" in immediate_ans["answer"]
    assert immediate_ans["verification"]["is_verified"] is True

    from pypdf import PdfWriter

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


def test_upgrade5_pdfplumber_sparse_empty_cells_and_nfd_unicode():
    """
    Kiểm tra lỗi thực tế khi `pdfplumber` sinh ra các ô trống `""` xen giữa cột Chỉ tiêu và Mã số VAS,
    hoặc khi PDF dùng bảng mã Unicode tổ hợp NFD.
    """
    extractor = FactExtractor()
    nfd_label = unicodedata.normalize("NFD", "1. Tiền và các khoản tương đương tiền")
    sparse_rows = [
        [nfd_label, "", None, "110", "", "V.02", "", "9.315.000", "", "8.279.000"],
    ]
    facts = extractor.extract_from_table_rows(
        sparse_rows, company="FPT", period="FY2024", statement_type="balance_sheet"
    )
    assert len(facts) == 1
    assert facts[0].vas_code == "110"
    assert facts[0].metric_code == "cash_and_equivalents"
    assert facts[0].value == 9315000.0


def test_upgrade6_wireless_plain_text_table_with_and_without_vas_code():
    """
    Kiểm tra hàm `_extract_vas_rows_from_plain_text` bóc tách được cả dòng có Mã số VAS
    lẫn dòng bảng tóm tắt không có cột Mã số VAS.
    """
    pipeline = UnifiedFinancialRAGPipeline()
    raw_page_text = (
        "I. TÀI SẢN NGẮN HẠN   100   V.01   45.610   36.705\n"
        "Tổng cộng tài sản   72.000   60.283\n"
    )
    rows = pipeline._extract_vas_rows_from_plain_text(raw_page_text)
    assert len(rows) == 2
    assert rows[0][1] == "100"
    assert rows[0][3] == "45.610"
    assert rows[1][1] == "270"
    assert rows[1][3] == "72.000"




def test_upgrade7_no_false_fpt_tagging_on_arbitrary_filenames():
    """
    Kiểm tra khắc phục triệt để lỗi mọi tài liệu nạp vào đều bị gắn nhãn FPT:
    1. Tên file có mã ở cuối (`Bao_cao_tai_chinh_Hop_nhat_2024_HPG.pdf`) -> phải nhận diện là `HPG`.
    2. Tên file không có mã ở đầu nhưng nội dung PDF ghi `CÔNG TY CỔ PHẦN SỮA VIỆT NAM` -> phải nhận diện là `VNM`.
    3. Tên file của một công ty bất kỳ ngoài danh sách (`BCTC_2024_Tap_doan_Hoa_Sen.pdf`) -> phải lấy tên từ file/nội dung,
       TUYỆT ĐỐI KHÔNG ĐƯỢC gán thành `FPT`.
    """
    pipeline = UnifiedFinancialRAGPipeline(seed_sample_data_if_empty=False)

    c1, p1 = pipeline.infer_company_and_period("Bao_cao_tai_chinh_Hop_nhat_2024_HPG.pdf")
    assert c1 == "HPG"
    assert p1 == "FY2024"

    c2, p2 = pipeline.infer_company_and_period(
        "BCTC_Hop_nhat_Q4_2024.pdf",
        first_pages_text="CÔNG TY CỔ PHẦN SỮA VIỆT NAM (VINAMILK) - BÁO CÁO TÀI CHÍNH HỢP NHẤT QUÝ 4 NĂM 2024",
    )
    assert c2 == "VNM"
    assert p2 == "Q4_2024"

    c3, p3 = pipeline.infer_company_and_period(
        "20240330_BCTC_2023.pdf",
        first_pages_text="CÔNG TY CỔ PHẦN TẬP ĐOÀN HOA SEN - BÁO CÁO TÀI CHÍNH HỢP NHẤT NĂM 2023",
    )
    assert c3 != "FPT"
    assert "TAP_DOAN_HOA" in c3 or "HOA_SEN" in c3


class TestUnifiedFinancialPipelineSuite(unittest.TestCase):
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

    def test_07_unit_scale_normalizer(self):
        test_upgrade1_unit_scale_normalizer()

    def test_08_report_scope_detector(self):
        test_upgrade2_report_scope_detector()

    def test_09_vlm_ocr_sqlite_cache(self):
        test_upgrade3_vlm_ocr_sqlite_cache()

    def test_10_default_seed_and_real_pdf_file_ingestion(self):
        test_upgrade4_default_seed_and_real_pdf_file_ingestion()

    def test_11_pdfplumber_sparse_empty_cells_and_nfd_unicode(self):
        test_upgrade5_pdfplumber_sparse_empty_cells_and_nfd_unicode()

    def test_12_wireless_plain_text_table_with_and_without_vas_code(self):
        test_upgrade6_wireless_plain_text_table_with_and_without_vas_code()

    def test_13_no_false_fpt_tagging_on_arbitrary_filenames(self):
        test_upgrade7_no_false_fpt_tagging_on_arbitrary_filenames()


if __name__ == "__main__":
    unittest.main(verbosity=2)
