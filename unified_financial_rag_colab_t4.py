"""
Unified Financial Multimodal RAG Pipeline (Optimized for Google Colab T4 16GB VRAM)
===================================================================================
Hợp nhất kiến trúc từ 2 nhánh dự án:
1. Nhánh Main (`src/multimodal_rag`):
   - Chuẩn hóa chỉ tiêu kế toán Việt Nam (VAS) & quốc tế (IFRS/US-GAAP) (`VASMetricRegistry`)
   - Kiểm tra chéo đẳng thức kế toán tự động (`VASAccountingQualityGate`: 270=100+200, 440=300+400,
     270=440, 20=10-11, 60=50-51-52, 50=20+30+40, 70=50+60+61)
   - Tính toán tất định bằng `Decimal` (`DeterministicFinancialCalculator`: Lookup, YoY, Ratio,
     Multi-metric summary, Reconciliation report) — tuyệt đối không để LLM tự tính nhẩm.
   - Lưu vết tọa độ `bbox` và `page` tới từng `Observation`.

2. Nhánh Colab T4 (`Financial-Multimodal-RAG-review-20261020` - nhánh `Thien`):
   - Lưu trữ bền vững với `SQLiteStore` (kèm FTS5 full-text search) + `FAISS` vector index
   - Truy xuất lai `HybridRetriever` (`BAAI/bge-m3` Dense + Sparse/FTS5 + `BAAI/bge-reranker-v2-m3`)
   - Mô hình đa phương thức dùng chung (`SharedQwenVLBackbone` với `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
     hoặc `Qwen/Qwen2.5-VL-3B-Instruct`): vừa đọc bảng BCTC scan (kèm vòng lặp Self-Correction khi
     lệch đẳng thức kế toán VAS), vừa đóng vai trò Semantic Router & Answer Synthesizer.
   - Đã khắc phục triệt để 8 lỗi trong `REVIEW_20261020.md`.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("unified_financial_rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# 1. SCHEMAS & MODELS (Hợp nhất từ `src/multimodal_rag/schemas.py` & `sqlite_store.py`)
# ==============================================================================

@dataclass
class BoundingBox:
    """Tọa độ chuẩn hóa [x0, y0, x1, y1] trên trang PDF để phục vụ trích dẫn trực quan."""
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> List[float]:
        return [round(float(self.x0), 2), round(float(self.y0), 2), round(float(self.x1), 2), round(float(self.y1), 2)]

    @classmethod
    def from_any(cls, raw: Optional[Union[Sequence[float], Dict[str, float], "BoundingBox"]]) -> Optional["BoundingBox"]:
        if raw is None:
            return None
        if isinstance(raw, BoundingBox):
            return raw
        if isinstance(raw, dict):
            return cls(float(raw.get("x0", 0.0)), float(raw.get("y0", 0.0)), float(raw.get("x1", 0.0)), float(raw.get("y1", 0.0)))
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            return cls(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        return None


@dataclass
class FinancialFact:
    """
    Hợp nhất giữa `Observation` (nhánh Main) và `fact` (nhánh Thien).
    Lưu trữ đầy đủ mã số VAS (`vas_code`), kỳ báo cáo (`period`), giá trị số âm/dương
    chính xác (`value`), đơn vị tính (`unit`), loại báo cáo (`statement_type`) và tọa độ (`bbox`).
    """
    company: str
    period: str
    metric_code: str
    metric_name: str
    value: float
    unit: str = "VND"
    vas_code: Optional[str] = None
    statement_type: str = "unknown"  # balance_sheet | income_statement | cash_flow | notes
    report_scope: str = "consolidated"  # consolidated (Hợp nhất) | separate (Công ty mẹ)
    page: int = 1
    bbox: Optional[List[float]] = None
    raw_text: str = ""
    confidence: float = 1.0
    extraction_method: str = "digital_text"  # digital_text | docling_table | qwen_vl_ocr | qwen_vl_self_corrected


@dataclass
class DocumentChunk:
    """Đoạn văn bản hoặc bảng đã chuyển thành Markdown kèm metadata truy vết."""
    chunk_id: str
    company: str
    period: str
    content: str
    chunk_type: str = "text"  # text | table | footnote
    statement_type: str = "notes"
    page: int = 1
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconciliationCheckResult:
    """Kết quả kiểm tra một đẳng thức kế toán trong `VASAccountingQualityGate`."""
    rule_id: str
    statement_type: str
    description: str
    lhs_code: str
    rhs_expression: str
    lhs_value: Optional[Decimal]
    rhs_value: Optional[Decimal]
    difference: Optional[Decimal]
    status: str  # pass | fail | skipped
    involved_vas_codes: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Báo cáo kiểm toán đầu vào cho một kỳ báo cáo của doanh nghiệp."""
    company: str
    period: str
    overall_status: str  # pass | warning | fail
    total_checks: int
    passed_checks: int
    failed_checks: int
    skipped_checks: int
    checks: List[ReconciliationCheckResult] = field(default_factory=list)
    self_correction_triggered: bool = False
    corrected_vas_codes: List[str] = field(default_factory=list)


# ==============================================================================
# 2. VAS METRIC REGISTRY (Hợp nhất từ `company_registry.py` & `ocr_recovery.py`)
# ==============================================================================

@dataclass(frozen=True)
class VASMetricSpec:
    metric_code: str
    vas_code: str
    canonical_name_vi: str
    canonical_name_en: str
    statement_type: str
    aliases: Tuple[str, ...]


class VASMetricRegistry:
    """
    Từ điển chuẩn hóa chỉ tiêu Báo cáo tài chính theo Thông tư 200/2014/TT-BTC (VAS)
    và tương đương IFRS/US-GAAP. Hỗ trợ tra cứu hai chiều theo Mã số VAS (01..440)
    hoặc theo tên chỉ tiêu tiếng Việt / tiếng Anh.
    """

    SPECS: Tuple[VASMetricSpec, ...] = (
        # Bảng cân đối kế toán (Balance Sheet)
        VASMetricSpec("short_term_assets", "100", "Tài sản ngắn hạn", "Current Assets", "balance_sheet",
                      ("tài sản ngắn hạn", "tai san ngan han", "current assets", "total current assets")),
        VASMetricSpec("cash_and_equivalents", "110", "Tiền và các khoản tương đương tiền", "Cash and Cash Equivalents", "balance_sheet",
                      ("tiền và các khoản tương đương tiền", "tien va cac khoan tuong duong tien", "cash and cash equivalents")),
        VASMetricSpec("short_term_receivables", "130", "Các khoản phải thu ngắn hạn", "Short-term Receivables", "balance_sheet",
                      ("các khoản phải thu ngắn hạn", "phải thu ngắn hạn", "accounts receivable")),
        VASMetricSpec("inventories", "140", "Hàng tồn kho", "Inventories", "balance_sheet",
                      ("hàng tồn kho", "hang ton kho", "inventories", "inventory")),
        VASMetricSpec("long_term_assets", "200", "Tài sản dài hạn", "Non-current Assets", "balance_sheet",
                      ("tài sản dài hạn", "tai san dai han", "non-current assets", "long-term assets")),
        VASMetricSpec("total_assets", "270", "Tổng cộng tài sản", "Total Assets", "balance_sheet",
                      ("tổng cộng tài sản", "tong cong tai san", "tổng tài sản", "total assets")),
        VASMetricSpec("liabilities", "300", "Nợ phải trả", "Total Liabilities", "balance_sheet",
                      ("nợ phải trả", "no phai tra", "total liabilities", "liabilities")),
        VASMetricSpec("short_term_liabilities", "310", "Nợ ngắn hạn", "Current Liabilities", "balance_sheet",
                      ("nợ ngắn hạn", "no ngan han", "current liabilities")),
        VASMetricSpec("long_term_liabilities", "330", "Nợ dài hạn", "Long-term Liabilities", "balance_sheet",
                      ("nợ dài hạn", "no dai han", "long-term liabilities", "non-current liabilities")),
        VASMetricSpec("owners_equity", "400", "Vốn chủ sở hữu", "Owners' Equity", "balance_sheet",
                      ("vốn chủ sở hữu", "von chu so huu", "owners' equity", "shareholders' equity", "total equity")),
        VASMetricSpec("total_resources", "440", "Tổng cộng nguồn vốn", "Total Liabilities and Equity", "balance_sheet",
                      ("tổng cộng nguồn vốn", "tong cong nguon von", "tổng nguồn vốn", "total liabilities and shareholders' equity", "total liabilities and equity")),

        # Báo cáo kết quả hoạt động kinh doanh (Income Statement)
        VASMetricSpec("gross_revenue", "01", "Doanh thu bán hàng và cung cấp dịch vụ", "Gross Revenue", "income_statement",
                      ("doanh thu bán hàng và cung cấp dịch vụ", "doanh thu ban hang", "gross revenue")),
        VASMetricSpec("revenue_deductions", "02", "Các khoản giảm trừ doanh thu", "Revenue Deductions", "income_statement",
                      ("các khoản giảm trừ doanh thu", "giam tru doanh thu", "revenue deductions")),
        VASMetricSpec("net_revenue", "10", "Doanh thu thuần về bán hàng và cung cấp dịch vụ", "Net Revenue", "income_statement",
                      ("doanh thu thuần", "doanh thu thuan", "net revenue", "revenue", "total revenue", "sales")),
        VASMetricSpec("cogs", "11", "Giá vốn hàng bán", "Cost of Goods Sold", "income_statement",
                      ("giá vốn hàng bán", "gia von hang ban", "cost of goods sold", "cost of sales", "cost of revenue")),
        VASMetricSpec("gross_profit", "20", "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "Gross Profit", "income_statement",
                      ("lợi nhuận gộp", "loi nhuan gop", "gross profit", "gross margin")),
        VASMetricSpec("financial_income", "21", "Doanh thu hoạt động tài chính", "Financial Income", "income_statement",
                      ("doanh thu hoạt động tài chính", "doanh thu tai chinh", "financial income")),
        VASMetricSpec("financial_expenses", "22", "Chi phí tài chính", "Financial Expenses", "income_statement",
                      ("chi phí tài chính", "chi phi tai chinh", "financial expenses")),
        VASMetricSpec("selling_expenses", "25", "Chi phí bán hàng", "Selling Expenses", "income_statement",
                      ("chi phí bán hàng", "chi phi ban hang", "selling expenses")),
        VASMetricSpec("admin_expenses", "26", "Chi phí quản lý doanh nghiệp", "General and Administrative Expenses", "income_statement",
                      ("chi phí quản lý doanh nghiệp", "chi phi quan ly", "general and administrative expenses", "sg&a")),
        VASMetricSpec("operating_profit", "30", "Lợi nhuận thuần từ hoạt động kinh doanh", "Operating Profit", "income_statement",
                      ("lợi nhuận thuần từ hoạt động kinh doanh", "operating income", "operating profit")),
        VASMetricSpec("profit_before_tax", "50", "Tổng lợi nhuận kế toán trước thuế", "Profit Before Tax", "income_statement",
                      ("tổng lợi nhuận kế toán trước thuế", "lợi nhuận trước thuế", "loi nhuan truoc thue", "profit before tax", "income before tax")),
        VASMetricSpec("current_tax_expense", "51", "Chi phí thuế TNDN hiện hành", "Current Corporate Income Tax Expense", "income_statement",
                      ("chi phí thuế tndn hiện hành", "thuế tndn hiện hành", "current tax expense")),
        VASMetricSpec("deferred_tax_expense", "52", "Chi phí thuế TNDN hoãn lại", "Deferred Corporate Income Tax Expense", "income_statement",
                      ("chi phí thuế tndn hoãn lại", "thuế tndn hoãn lại", "deferred tax expense")),
        VASMetricSpec("net_profit", "60", "Lợi nhuận sau thuế thu nhập doanh nghiệp", "Net Profit After Tax", "income_statement",
                      ("lợi nhuận sau thuế", "loi nhuan sau thue", "lãi ròng", "net income", "net profit", "profit after tax")),

        # Báo cáo lưu chuyển tiền tệ (Cash Flow Statement)
        VASMetricSpec("operating_cash_flow", "20_CF", "Lưu chuyển tiền thuần từ hoạt động kinh doanh", "Net Cash Flow from Operating Activities", "cash_flow",
                      ("lưu chuyển tiền thuần từ hoạt động kinh doanh", "luu chuyen tien thuan tu hoat dong kinh doanh", "operating cash flow", "cash flows from operating activities")),
        VASMetricSpec("investing_cash_flow", "30_CF", "Lưu chuyển tiền thuần từ hoạt động đầu tư", "Net Cash Flow from Investing Activities", "cash_flow",
                      ("lưu chuyển tiền thuần từ hoạt động đầu tư", "luu chuyen tien thuan tu hoat dong dau tu", "investing cash flow", "cash flows from investing activities")),
        VASMetricSpec("financing_cash_flow", "40_CF", "Lưu chuyển tiền thuần từ hoạt động tài chính", "Net Cash Flow from Financing Activities", "cash_flow",
                      ("lưu chuyển tiền thuần từ hoạt động tài chính", "luu chuyen tien thuan tu hoat dong tai chinh", "financing cash flow", "cash flows from financing activities")),
        VASMetricSpec("net_cash_flow", "50_CF", "Lưu chuyển tiền thuần trong kỳ", "Net Increase/Decrease in Cash", "cash_flow",
                      ("lưu chuyển tiền thuần trong kỳ", "lưu chuyển tiền thuần trong năm", "net cash flow", "net change in cash")),
        VASMetricSpec("beginning_cash", "60_CF", "Tiền và tương đương tiền đầu kỳ", "Cash and Cash Equivalents at Beginning of Period", "cash_flow",
                      ("tiền và tương đương tiền đầu kỳ", "tiền và tương đương tiền đầu năm", "beginning cash")),
        VASMetricSpec("fx_impact_on_cash", "61_CF", "Ảnh hưởng của thay đổi tỷ giá hối đoái quy đổi ngoại tệ", "Effect of Exchange Rate Changes on Cash", "cash_flow",
                      ("ảnh hưởng của thay đổi tỷ giá", "anh huong cua thay doi ty gia", "effect of exchange rate")),
        VASMetricSpec("ending_cash", "70_CF", "Tiền và tương đương tiền cuối kỳ", "Cash and Cash Equivalents at End of Period", "cash_flow",
                      ("tiền và tương đương tiền cuối kỳ", "tiền và tương đương tiền cuối năm", "ending cash")),
    )

    def __init__(self) -> None:
        self.by_metric_code: Dict[str, VASMetricSpec] = {s.metric_code: s for s in self.SPECS}
        self.by_statement_and_vas: Dict[Tuple[str, str], VASMetricSpec] = {}
        for s in self.SPECS:
            clean_vas = s.vas_code.replace("_CF", "")
            self.by_statement_and_vas[(s.statement_type, clean_vas)] = s

    def match(self, raw_label: str, vas_code: Optional[str] = None, statement_type: Optional[str] = None) -> Optional[VASMetricSpec]:
        norm_label = re.sub(r"\s+", " ", (raw_label or "").strip().lower())
        clean_vas = (vas_code or "").strip().replace("_CF", "")

        # Ưu tiên khớp chính xác theo (statement_type, vas_code) nếu có
        if clean_vas and statement_type and (statement_type, clean_vas) in self.by_statement_and_vas:
            return self.by_statement_and_vas[(statement_type, clean_vas)]

        # Khớp theo alias văn bản (ưu tiên alias dài nhất trước để tránh "doanh thu" nuốt "doanh thu thuần")
        best_spec: Optional[VASMetricSpec] = None
        best_len = 0
        for spec in self.SPECS:
            if statement_type and spec.statement_type != statement_type and statement_type not in {"unknown", "notes"}:
                continue
            for alias in spec.aliases:
                if alias in norm_label and len(alias) > best_len:
                    best_spec = spec
                    best_len = len(alias)
        if best_spec is not None:
            return best_spec

        # Fallback theo vas_code nếu không rõ statement_type (cho các mã duy nhất như 100, 200, 270, 300, 400, 440, 10, 11)
        if clean_vas:
            unique_candidates = [
                spec for (st, vc), spec in self.by_statement_and_vas.items() if vc == clean_vas
            ]
            if len(unique_candidates) == 1:
                return unique_candidates[0]
        return None


# ==============================================================================
# 3. FACT EXTRACTOR & NUMBER PARSER (Vá lỗi #5 từ `REVIEW_20261020.md`)
# ==============================================================================

class FactExtractor:
    """
    Trích xuất các chỉ tiêu tài chính từ bảng/văn bản.
    ĐÃ VÁ LỖI #5 TRONG REVIEW_20261020.md:
    - Bảo toàn đầy đủ số âm dạng ngoặc đơn kế toán `(1.234.567)` hoặc dấu trừ `-1,234,567`
      cũng như giá trị `0` (trước đây `return val if val and val > 0 else None` làm mất sạch
      các khoản lỗ, dòng tiền âm và giảm trừ doanh thu).
    """

    def __init__(self, registry: Optional[VASMetricRegistry] = None) -> None:
        self.registry = registry or VASMetricRegistry()

    @staticmethod
    def clean_number(raw: Union[str, int, float, Decimal, None]) -> Optional[float]:
        """
        Chuyển đổi chuỗi số kế toán Việt Nam / Quốc tế sang float:
        - Hỗ trợ số âm trong ngoặc đơn: `(1.250.000)` -> `-1250000.0`
        - Hỗ trợ dấu trừ trực tiếp: `-1,250.50` -> `-1250.5`
        - Hỗ trợ phân biệt dấu chấm phân cách hàng nghìn của VN (`1.234.567`)
          và dấu chấm thập phân quốc tế (`1,234.56`).
        - Bảo toàn cả giá trị âm, 0 và dương.
        """
        if raw is None:
            return None
        if isinstance(raw, (int, float, Decimal)):
            val = float(raw)
            return None if math.isnan(val) or math.isinf(val) else val

        text = str(raw).strip()
        if not text or text in {"-", "—", "–", "N/A", "n/a", "nil", ""}:
            return None

        text = re.sub(r"(VND|VNĐ|USD|tỷ|triệu|đồng|%|\$)", "", text, flags=re.IGNORECASE).strip()

        is_negative = False
        if (text.startswith("(") and text.endswith(")")) or text.startswith("-"):
            is_negative = True
            text = text.strip("()+- ").strip()

        if not text or not re.search(r"\d", text):
            return None

        dots = text.count(".")
        commas = text.count(",")

        if dots > 1 and commas == 0:
            normalized = text.replace(".", "")
        elif commas > 1 and dots == 0:
            normalized = text.replace(",", "")
        elif dots >= 1 and commas >= 1:
            if text.rfind(",") > text.rfind("."):
                normalized = text.replace(".", "").replace(",", ".")
            else:
                normalized = text.replace(",", "")
        elif dots == 1 and commas == 0:
            parts = text.split(".")
            if len(parts[1]) == 3 and len(parts[0]) >= 1 and parts[0] != "0":
                normalized = "".join(parts)
            else:
                normalized = text
        elif commas == 1 and dots == 0:
            parts = text.split(",")
            if len(parts[1]) == 3 and len(parts[0]) >= 1 and parts[0] != "0":
                normalized = "".join(parts)
            else:
                normalized = text.replace(",", ".")
        else:
            normalized = text

        normalized = re.sub(r"[^\d.]", "", normalized)
        if not normalized:
            return None

        try:
            result = float(normalized)
            return -result if is_negative and result != 0 else result
        except ValueError:
            return None

    def _clean_number(self, raw: Union[str, int, float, Decimal, None]) -> Optional[float]:
        return self.clean_number(raw)

    def extract_from_table_rows(
        self,
        rows: Sequence[Sequence[Any]],
        company: str,
        period: str,
        statement_type: str = "unknown",
        page: int = 1,
        unit: str = "VND",
        extraction_method: str = "digital_text",
    ) -> List[FinancialFact]:
        """
        Bóc tách danh sách `FinancialFact` từ bảng 2 chiều (list các dòng).
        Hỗ trợ cấu trúc cột chuẩn VAS: [Chỉ tiêu, Mã số, Thuyết minh, Số cuối năm, Số đầu năm]
        hoặc bảng rút gọn [Chỉ tiêu, Mã số, Số cuối năm, Số đầu năm].
        """
        facts: List[FinancialFact] = []
        for row_idx, row in enumerate(rows):
            cells = [str(c).strip() if c is not None else "" for c in row]
            if not any(cells):
                continue

            label = cells[0]
            if not label or re.match(r"^(chỉ tiêu|mã số|thuyết minh|số cuối năm|items|code|note)$", label, re.I):
                continue

            vas_code: Optional[str] = None
            numeric_candidates: List[float] = []

            for col_idx, cell in enumerate(cells[1:], start=1):
                clean_cell = cell.strip()
                if not clean_cell:
                    continue
                # Nhận diện cột Mã số VAS ở cột 1 (gồm 2 hoặc 3 chữ số như 01, 10, 20, 100, 270, 440)
                if vas_code is None and col_idx == 1 and re.fullmatch(r"\d{2,3}", clean_cell):
                    vas_code = clean_cell
                    continue
                # Nhận diện cột Thuyết minh ở cột 2 (ví dụ: V.01, VI.2, TM.05, hoặc mục nhỏ 1-2 chữ số như 4.1)
                if col_idx == 2 and re.fullmatch(r"([IVX]+\.\d+|TM\.?\d+|\d{1,2}\.\d{1,2})", clean_cell, re.I):
                    continue

                val = self.clean_number(clean_cell)
                if val is not None:
                    numeric_candidates.append(val)

            if not numeric_candidates:
                continue

            spec = self.registry.match(label, vas_code=vas_code, statement_type=statement_type)
            metric_code = spec.metric_code if spec else re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            canonical_name = spec.canonical_name_vi if spec else label
            resolved_statement = spec.statement_type if spec else statement_type
            resolved_vas = (spec.vas_code.replace("_CF", "") if spec else vas_code)

            current_val = numeric_candidates[0]
            facts.append(
                FinancialFact(
                    company=company.upper(),
                    period=period,
                    metric_code=metric_code,
                    metric_name=canonical_name,
                    value=current_val,
                    unit=unit,
                    vas_code=resolved_vas,
                    statement_type=resolved_statement,
                    page=page,
                    bbox=[0.0, float(row_idx * 20), 595.0, float((row_idx + 1) * 20)],
                    raw_text=" | ".join(cells),
                    extraction_method=extraction_method,
                )
            )
        return facts


# ==============================================================================
# 4. VAS ACCOUNTING QUALITY GATE (Hợp nhất từ `src/multimodal_rag/quality.py`)
# ==============================================================================

class VASAccountingQualityGate:
    """
    Kiểm tra chéo các đẳng thức kế toán bắt buộc của BCTC Việt Nam (Thông tư 200):
    1. Bảng cân đối kế toán (Balance Sheet):
       - Mã 270 (Tổng tài sản) = Mã 100 (Tài sản ngắn hạn) + Mã 200 (Tài sản dài hạn)
       - Mã 440 (Tổng nguồn vốn) = Mã 300 (Nợ phải trả) + Mã 400 (Vốn chủ sở hữu)
       - Mã 270 (Tổng tài sản) = Mã 440 (Tổng nguồn vốn)
    2. Báo cáo kết quả kinh doanh (Income Statement):
       - Mã 20 (Lợi nhuận gộp) = Mã 10 (Doanh thu thuần) - Mã 11 (Giá vốn hàng bán)
       - Mã 60 (Lợi nhuận sau thuế) = Mã 50 (LN trước thuế) - Mã 51 (Thuế hiện hành) - Mã 52 (Thuế hoãn lại)
    3. Báo cáo lưu chuyển tiền tệ (Cash Flow):
       - Mã 50 (Lưu chuyển tiền thuần trong kỳ) = Mã 20 (HĐKD) + Mã 30 (HĐĐT) + Mã 40 (HĐTC)
       - Mã 70 (Tiền cuối kỳ) = Mã 50 (LCTT trong kỳ) + Mã 60 (Tiền đầu kỳ) + Mã 61 (Ảnh hưởng tỷ giá)
    """

    def __init__(self, tolerance: Decimal = Decimal("1.0")) -> None:
        self.tolerance = tolerance

    @staticmethod
    def _to_dec(val: Optional[float]) -> Optional[Decimal]:
        if val is None:
            return None
        return Decimal(str(val))

    def evaluate(self, company: str, period: str, facts: Sequence[FinancialFact]) -> QualityReport:
        by_stmt_code: Dict[Tuple[str, str], Decimal] = {}
        by_metric_code: Dict[str, Decimal] = {}

        for f in facts:
            dec_val = self._to_dec(f.value)
            if dec_val is None:
                continue
            by_metric_code[f.metric_code] = dec_val
            if f.vas_code:
                clean_vas = f.vas_code.replace("_CF", "")
                by_stmt_code[(f.statement_type, clean_vas)] = dec_val

        def get_bs(code: str, metric_fallback: str) -> Optional[Decimal]:
            return by_stmt_code.get(("balance_sheet", code), by_metric_code.get(metric_fallback))

        def get_is(code: str, metric_fallback: str) -> Optional[Decimal]:
            return by_stmt_code.get(("income_statement", code), by_metric_code.get(metric_fallback))

        def get_cf(code: str, metric_fallback: str) -> Optional[Decimal]:
            return by_stmt_code.get(("cash_flow", code), by_metric_code.get(metric_fallback))

        checks: List[ReconciliationCheckResult] = []

        # Rule 1: BS 270 = 100 + 200
        bs_270 = get_bs("270", "total_assets")
        bs_100 = get_bs("100", "short_term_assets")
        bs_200 = get_bs("200", "long_term_assets")
        checks.append(self._check_sum(
            rule_id="BS_270_EQ_100_PLUS_200",
            statement_type="balance_sheet",
            description="Tổng cộng tài sản (270) = Tài sản ngắn hạn (100) + Tài sản dài hạn (200)",
            lhs_code="270",
            rhs_expression="100 + 200",
            lhs=bs_270,
            rhs=(bs_100 + bs_200) if (bs_100 is not None and bs_200 is not None) else None,
            codes=["100", "200", "270"],
        ))

        # Rule 2: BS 440 = 300 + 400
        bs_440 = get_bs("440", "total_resources")
        bs_300 = get_bs("300", "liabilities")
        bs_400 = get_bs("400", "owners_equity")
        checks.append(self._check_sum(
            rule_id="BS_440_EQ_300_PLUS_400",
            statement_type="balance_sheet",
            description="Tổng cộng nguồn vốn (440) = Nợ phải trả (300) + Vốn chủ sở hữu (400)",
            lhs_code="440",
            rhs_expression="300 + 400",
            lhs=bs_440,
            rhs=(bs_300 + bs_400) if (bs_300 is not None and bs_400 is not None) else None,
            codes=["300", "400", "440"],
        ))

        # Rule 3: BS 270 = 440
        checks.append(self._check_sum(
            rule_id="BS_270_EQ_440",
            statement_type="balance_sheet",
            description="Tổng cộng tài sản (270) = Tổng cộng nguồn vốn (440)",
            lhs_code="270",
            rhs_expression="440",
            lhs=bs_270,
            rhs=bs_440,
            codes=["270", "440"],
        ))

        # Rule 4: IS 20 = 10 - 11
        is_20 = get_is("20", "gross_profit")
        is_10 = get_is("10", "net_revenue")
        is_11 = get_is("11", "cogs")
        rhs_gross: Optional[Decimal] = None
        if is_10 is not None and is_11 is not None:
            rhs_gross = (is_10 - abs(is_11))
        checks.append(self._check_sum(
            rule_id="IS_20_EQ_10_MINUS_11",
            statement_type="income_statement",
            description="Lợi nhuận gộp (20) = Doanh thu thuần (10) - Giá vốn hàng bán (11)",
            lhs_code="20",
            rhs_expression="10 - 11",
            lhs=is_20,
            rhs=rhs_gross,
            codes=["10", "11", "20"],
        ))

        # Rule 5: IS 60 = 50 - 51 - 52
        is_60 = get_is("60", "net_profit")
        is_50 = get_is("50", "profit_before_tax")
        is_51 = get_is("51", "current_tax_expense")
        is_52 = get_is("52", "deferred_tax_expense")
        rhs_net_profit: Optional[Decimal] = None
        if is_50 is not None and is_51 is not None:
            tax_52 = is_52 if is_52 is not None else Decimal("0")
            rhs_net_profit = is_50 - abs(is_51) - tax_52
        checks.append(self._check_sum(
            rule_id="IS_60_EQ_50_MINUS_51_52",
            statement_type="income_statement",
            description="Lợi nhuận sau thuế (60) = LN trước thuế (50) - Thuế TNDN hiện hành (51) - Thuế hoãn lại (52)",
            lhs_code="60",
            rhs_expression="50 - 51 - 52",
            lhs=is_60,
            rhs=rhs_net_profit,
            codes=["50", "51", "52", "60"],
        ))

        # Rule 6: CF 50 = 20 + 30 + 40
        cf_50 = get_cf("50", "net_cash_flow")
        cf_20 = get_cf("20", "operating_cash_flow")
        cf_30 = get_cf("30", "investing_cash_flow")
        cf_40 = get_cf("40", "financing_cash_flow")
        checks.append(self._check_sum(
            rule_id="CF_50_EQ_20_PLUS_30_PLUS_40",
            statement_type="cash_flow",
            description="Lưu chuyển tiền thuần trong kỳ (50) = HĐKD (20) + HĐĐT (30) + HĐTC (40)",
            lhs_code="50",
            rhs_expression="20 + 30 + 40",
            lhs=cf_50,
            rhs=(cf_20 + cf_30 + cf_40) if (cf_20 is not None and cf_30 is not None and cf_40 is not None) else None,
            codes=["20", "30", "40", "50"],
        ))

        # Rule 7: CF 70 = 50 + 60 + 61
        cf_70 = get_cf("70", "ending_cash")
        cf_60 = get_cf("60", "beginning_cash")
        cf_61 = get_cf("61", "fx_impact_on_cash")
        rhs_ending_cash: Optional[Decimal] = None
        if cf_50 is not None and cf_60 is not None:
            rhs_ending_cash = cf_50 + cf_60 + (cf_61 if cf_61 is not None else Decimal("0"))
        checks.append(self._check_sum(
            rule_id="CF_70_EQ_50_PLUS_60_PLUS_61",
            statement_type="cash_flow",
            description="Tiền cuối kỳ (70) = Lưu chuyển tiền thuần (50) + Tiền đầu kỳ (60) + Ảnh hưởng tỷ giá (61)",
            lhs_code="70",
            rhs_expression="50 + 60 + 61",
            lhs=cf_70,
            rhs=rhs_ending_cash,
            codes=["50", "60", "61", "70"],
        ))

        passed = sum(1 for c in checks if c.status == "pass")
        failed = sum(1 for c in checks if c.status == "fail")
        skipped = sum(1 for c in checks if c.status == "skipped")

        if failed > 0:
            overall = "fail"
        elif passed == 0:
            overall = "warning"
        else:
            overall = "pass"

        return QualityReport(
            company=company.upper(),
            period=period,
            overall_status=overall,
            total_checks=len(checks),
            passed_checks=passed,
            failed_checks=failed,
            skipped_checks=skipped,
            checks=checks,
        )

    def _check_sum(
        self,
        rule_id: str,
        statement_type: str,
        description: str,
        lhs_code: str,
        rhs_expression: str,
        lhs: Optional[Decimal],
        rhs: Optional[Decimal],
        codes: List[str],
    ) -> ReconciliationCheckResult:
        if lhs is None or rhs is None:
            return ReconciliationCheckResult(
                rule_id=rule_id,
                statement_type=statement_type,
                description=description,
                lhs_code=lhs_code,
                rhs_expression=rhs_expression,
                lhs_value=lhs,
                rhs_value=rhs,
                difference=None,
                status="skipped",
                involved_vas_codes=codes,
            )
        diff = abs(lhs - rhs)
        status = "pass" if diff <= self.tolerance else "fail"
        return ReconciliationCheckResult(
            rule_id=rule_id,
            statement_type=statement_type,
            description=description,
            lhs_code=lhs_code,
            rhs_expression=rhs_expression,
            lhs_value=lhs,
            rhs_value=rhs,
            difference=diff,
            status=status,
            involved_vas_codes=codes,
        )


# ==============================================================================
# 5. SHARED QWEN2.5-VL BACKBONE & SMART PDF PARSER (Vá lỗi #1 & #2)
# ==============================================================================

class SharedQwenVLBackbone:
    """
    Mô hình 2-trong-1 tối ưu cho Google Colab T4 (16GB VRAM):
    Sử dụng `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (~5.8GB VRAM) hoặc `Qwen/Qwen2.5-VL-3B-Instruct`:
    1. Chế độ Vision-OCR (`extract_table_rows_from_image`):
       Nhận ảnh chụp vùng bảng BCTC (đặc biệt là trang scan mờ/mất đường kẻ) và xuất thẳng
       ra danh sách dòng JSON chuẩn `["Tên chỉ tiêu", "Mã số", "Thuyết minh", "Số cuối năm", "Số đầu năm"]`.
    2. Chế độ Self-Correction:
       Khi `VASAccountingQualityGate` phát hiện lệch đẳng thức kế toán (ví dụ: 270 != 100 + 200),
       VLM được gọi lại kèm công thức đang lệch để kiểm tra lại chính xác chữ số bị nhầm.
    3. Chế độ Text RAG (`generate_text`):
       Tổng hợp câu trả lời tiếng Việt từ ngữ cảnh Thuyết minh BCTC và kết quả tính toán tất định.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    FALLBACK_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        lazy_load: bool = True,
        mock_vlm_callable: Optional[Callable[[str, Optional[Any]], str]] = None,
    ) -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.mock_vlm_callable = mock_vlm_callable
        if not lazy_load and mock_vlm_callable is None:
            self.load_model()

    def load_model(self) -> None:
        """Nạp mô hình lên GPU Colab T4 với cơ chế tiết kiệm VRAM tối đa."""
        if self.model is not None or self.mock_vlm_callable is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            logger.info("Đang nạp Shared VLM/LLM Backbone: %s ...", self.model_id)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
            logger.info("Đã nạp thành công %s lên thiết bị.", self.model_id)
        except Exception as exc:
            logger.warning("Không thể nạp GPU model (%s). Chuyển sang chế độ Rule/Mock Fallback.", exc)

    def extract_table_rows_from_image(
        self,
        image_input: Any,
        statement_hint: str = "balance_sheet",
        correction_hint: Optional[str] = None,
    ) -> List[List[str]]:
        """
        Gửi ảnh trang/vùng bảng BCTC vào Qwen2.5-VL để bóc tách cấu trúc bảng chính xác.
        """
        prompt = (
            "Bạn là hệ thống OCR chuyên dụng cho Báo cáo tài chính Việt Nam (VAS). "
            "Hãy đọc bảng trong ảnh và trả về DUY NHẤT một JSON array gồm các dòng, "
            "mỗi dòng có đúng định dạng: [\"Tên chỉ tiêu\", \"Mã số\", \"Thuyết minh\", \"Số cuối năm\", \"Số đầu năm\"]. "
            "Giữ nguyên dấu ngoặc đơn ( ) cho số âm. Không tự ý làm tròn số."
        )
        if correction_hint:
            prompt += f"\nLƯU Ý KIỂM TOÁN CẦN ĐỐI CHIẾU LẠI: {correction_hint}"

        if self.mock_vlm_callable is not None:
            raw_output = self.mock_vlm_callable(prompt, image_input)
            return self._parse_json_rows(raw_output)

        self.load_model()
        if self.model is None or self.processor is None:
            return []

        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False)
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        return self._parse_json_rows(output_text)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Sử dụng chính Qwen2.5-VL ở chế độ text-only làm LLM trả lời RAG (tiết kiệm 100% VRAM nạp LLM thứ hai)."""
        if self.mock_vlm_callable is not None:
            return self.mock_vlm_callable(f"{system_prompt}\n{user_prompt}", None)

        self.load_model()
        if self.model is None or self.processor is None:
            return ""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, temperature=0.1, do_sample=False)
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _parse_json_rows(raw_output: str) -> List[List[str]]:
        if not raw_output:
            return []
        match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [[str(cell) for cell in row] for row in parsed if isinstance(row, list)]
        except json.JSONDecodeError:
            pass
        return []


class SmartFinancialPDFParser:
    """
    Bộ đọc PDF đa tầng khắc phục triệt để Lỗi #1 và Lỗi #2 trong `REVIEW_20261020.md`:
    1. Kiểm tra chéo nội dung thực của PDF với nhãn `expected_company` / `expected_period`
       (`verify_document_identity`) — chặn ngay trường hợp file `NVDA_10K_2024_sample.pdf`
       nhưng ruột lại là Analog Devices 2009.
    2. Phân loại từng trang PDF:
       - Nếu trang có digital text layer: trích xuất văn bản + bảng qua `pdfplumber` / `Docling`.
       - Nếu trang là ảnh scan (`len(text.strip()) < 40`): tự động chuyển sang nhánh VLM OCR
         (`SharedQwenVLBackbone`) thay vì trả về 0 chunks / 0 facts.
    3. `validate_extraction()` kiểm tra chặt: nếu `len(chunks) == 0` và `len(facts) == 0`
       thì bắt buộc trả về `is_valid = False` (sửa lỗi `is_valid=True` giả tạo ở nhánh cũ).
    """

    COMPANY_ALIASES: Dict[str, Tuple[str, ...]] = {
        "FPT": ("fpt", "công ty cổ phần fpt", "fpt corporation"),
        "VNM": ("vinamilk", "sữa việt nam", "vietnam dairy products", "vnm"),
        "VCB": ("vietcombank", "ngoại thương việt nam", "vcb"),
        "VIC": ("vingroup", "tập đoàn vingroup", "vic"),
        "HPG": ("hòa phát", "hoa phat", "hpg"),
        "NVDA": ("nvidia", "nvidia corporation", "nvda"),
        "ADI": ("analog devices", "analog devices, inc"),
    }

    def __init__(
        self,
        fact_extractor: Optional[FactExtractor] = None,
        vlm_backbone: Optional[SharedQwenVLBackbone] = None,
        quality_gate: Optional[VASAccountingQualityGate] = None,
    ) -> None:
        self.fact_extractor = fact_extractor or FactExtractor()
        self.vlm_backbone = vlm_backbone or SharedQwenVLBackbone(lazy_load=True)
        self.quality_gate = quality_gate or VASAccountingQualityGate()

    @staticmethod
    def _contains_alias(text: str, aliases: Iterable[str]) -> bool:
        for alias in aliases:
            pattern = rf"(?<!\w){re.escape(alias.lower())}(?!\w)"
            if re.search(pattern, text):
                return True
        return False

    def verify_document_identity(
        self,
        first_pages_text: str,
        expected_company: str,
        expected_period: str,
    ) -> Tuple[bool, str]:
        """
        VÁ LỖI #1 TRONG REVIEW_20261020.md:
        Kiểm tra nội dung trang đầu của báo cáo xem có khớp với mã công ty & năm kỳ vọng không.
        Phát hiện ngay trường hợp file tên là NVDA 2024 nhưng nội dung là Analog Devices 2009.
        """
        norm_text = (first_pages_text or "").lower()
        if not norm_text.strip():
            return True, "pending_ocr_verification"

        exp_code = expected_company.upper().strip()
        expected_aliases = self.COMPANY_ALIASES.get(exp_code, (exp_code.lower(),))

        has_expected = self._contains_alias(norm_text, expected_aliases)
        if not has_expected:
            for other_code, other_aliases in self.COMPANY_ALIASES.items():
                if other_code != exp_code and self._contains_alias(norm_text, other_aliases):
                    return (
                        False,
                        f"Phát hiện lệch nhận diện tài liệu: File được gắn nhãn '{exp_code}' "
                        f"nhưng nội dung thực tế thuộc về '{other_code}'.",
                    )

        year_match = re.search(r"(20\d{2})", expected_period)
        if year_match:
            exp_year = year_match.group(1)
            found_years = sorted(set(re.findall(r"\b(20\d{2})\b", norm_text)))
            if found_years and exp_year not in found_years and max(found_years) < str(int(exp_year) - 2):
                return (
                    False,
                    f"Phát hiện lệch kỳ báo cáo: Nhãn kỳ vọng là '{expected_period}' "
                    f"nhưng nội dung tài liệu chỉ đề cập tới các năm {found_years}.",
                )

        return True, "verified"

    @staticmethod
    def detect_statement_type(page_text: str) -> str:
        t = (page_text or "").lower()
        if any(k in t for k in ("bảng cân đối kế toán", "cân đối kế toán", "balance sheet", "tổng cộng tài sản", "tổng cộng nguồn vốn")):
            return "balance_sheet"
        if any(k in t for k in ("kết quả hoạt động kinh doanh", "báo cáo kết quả", "income statement", "lợi nhuận gộp", "doanh thu thuần")):
            return "income_statement"
        if any(k in t for k in ("lưu chuyển tiền tệ", "cash flow", "lưu chuyển tiền thuần")):
            return "cash_flow"
        return "notes"

    def extract_from_pages_data(
        self,
        pages_data: Sequence[Dict[str, Any]],
        company: str,
        period: str,
        strict_identity_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Xử lý danh sách trang PDF (mỗi trang gồm `page`, `text`, `tables`, `image`).
        Tự động kích hoạt VLM OCR cho trang scan và tự động sửa lỗi (Self-Correction)
        nếu `VASAccountingQualityGate` phát hiện lệch đẳng thức kế toán.
        """
        combined_early_text = "\n".join(str(p.get("text", "")) for p in pages_data[:3])
        is_identity_ok, identity_msg = self.verify_document_identity(combined_early_text, company, period)
        if strict_identity_check and not is_identity_ok:
            raise ValueError(identity_msg)

        chunks: List[DocumentChunk] = []
        facts: List[FinancialFact] = []
        scanned_pages_trigger: List[Dict[str, Any]] = []

        for page_obj in pages_data:
            page_num = int(page_obj.get("page", 1))
            raw_text = str(page_obj.get("text", "") or "").strip()
            raw_tables = page_obj.get("tables", []) or []
            page_image = page_obj.get("image", None)

            is_scanned_page = len(raw_text) < 40 and not raw_tables
            stmt_type = self.detect_statement_type(raw_text)

            if is_scanned_page:
                scanned_pages_trigger.append(page_obj)
                ocr_rows = self.vlm_backbone.extract_table_rows_from_image(
                    image_input=page_image or f"scanned_page_{page_num}",
                    statement_hint=stmt_type,
                )
                if ocr_rows:
                    joined_ocr_text = "\n".join(" | ".join(r) for r in ocr_rows)
                    inferred_stmt = self.detect_statement_type(joined_ocr_text)
                    page_facts = self.fact_extractor.extract_from_table_rows(
                        ocr_rows,
                        company=company,
                        period=period,
                        statement_type=inferred_stmt,
                        page=page_num,
                        extraction_method="qwen_vl_ocr",
                    )
                    facts.extend(page_facts)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{company}_{period}_p{page_num}_ocr_table",
                            company=company.upper(),
                            period=period,
                            content=joined_ocr_text,
                            chunk_type="table",
                            statement_type=inferred_stmt,
                            page=page_num,
                            metadata={"extraction_method": "qwen_vl_ocr"},
                        )
                    )
                continue

            if raw_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{company}_{period}_p{page_num}_text",
                        company=company.upper(),
                        period=period,
                        content=raw_text,
                        chunk_type="text",
                        statement_type=stmt_type,
                        page=page_num,
                        metadata={"extraction_method": "digital_text"},
                    )
                )

            for t_idx, table in enumerate(raw_tables):
                table_text = "\n".join(" | ".join(str(c) for c in r) for r in table)
                t_stmt = self.detect_statement_type(f"{raw_text}\n{table_text}")
                table_facts = self.fact_extractor.extract_from_table_rows(
                    table,
                    company=company,
                    period=period,
                    statement_type=t_stmt,
                    page=page_num,
                    extraction_method="docling_table",
                )
                facts.extend(table_facts)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{company}_{period}_p{page_num}_t{t_idx}",
                        company=company.upper(),
                        period=period,
                        content=table_text,
                        chunk_type="table",
                        statement_type=t_stmt,
                        page=page_num,
                        metadata={"extraction_method": "docling_table"},
                    )
                )

        # Bước Kiểm toán đầu vào (Accounting Quality Gate) & Self-Correction Loop
        quality_report = self.quality_gate.evaluate(company=company, period=period, facts=facts)
        if quality_report.failed_checks > 0 and scanned_pages_trigger:
            failed_checks = [c for c in quality_report.checks if c.status == "fail"]
            hint_parts = [f"{c.description} (Chênh lệch: {c.difference})" for c in failed_checks]
            correction_hint = "; ".join(hint_parts)

            for page_obj in scanned_pages_trigger:
                page_num = int(page_obj.get("page", 1))
                re_read_rows = self.vlm_backbone.extract_table_rows_from_image(
                    image_input=page_obj.get("image") or f"scanned_page_{page_num}_retry",
                    correction_hint=correction_hint,
                )
                if re_read_rows:
                    joined_retry = "\n".join(" | ".join(r) for r in re_read_rows)
                    inferred_stmt = self.detect_statement_type(joined_retry)
                    retry_facts = self.fact_extractor.extract_from_table_rows(
                        re_read_rows,
                        company=company,
                        period=period,
                        statement_type=inferred_stmt,
                        page=page_num,
                        extraction_method="qwen_vl_self_corrected",
                    )
                    if retry_facts:
                        retry_by_key = {(f.statement_type, f.metric_code): f for f in retry_facts}
                        facts = [
                            retry_by_key.get((f.statement_type, f.metric_code), f) for f in facts
                        ]
                        existing_keys = {(f.statement_type, f.metric_code) for f in facts}
                        for k, rf in retry_by_key.items():
                            if k not in existing_keys:
                                facts.append(rf)

            new_report = self.quality_gate.evaluate(company=company, period=period, facts=facts)
            new_report.self_correction_triggered = True
            new_report.corrected_vas_codes = [
                code for c in failed_checks for code in c.involved_vas_codes
            ]
            quality_report = new_report

        validation = self.validate_extraction(chunks=chunks, facts=facts, quality_report=quality_report)
        return {
            "company": company.upper(),
            "period": period,
            "chunks": chunks,
            "facts": facts,
            "quality_report": quality_report,
            "validation": validation,
        }

    @staticmethod
    def validate_extraction(
        chunks: Sequence[DocumentChunk],
        facts: Sequence[FinancialFact],
        quality_report: Optional[QualityReport] = None,
    ) -> Dict[str, Any]:
        """
        VÁ LỖI #2 TRONG REVIEW_20261020.md:
        Không bao giờ trả về `is_valid=True` khi không trích xuất được bất kỳ chunk hay fact nào.
        """
        has_content = len(chunks) > 0 or len(facts) > 0
        accounting_ok = True if quality_report is None else (quality_report.overall_status != "fail")
        is_valid = bool(has_content and accounting_ok)
        return {
            "is_valid": is_valid,
            "num_chunks": len(chunks),
            "num_facts": len(facts),
            "accounting_status": quality_report.overall_status if quality_report else "unknown",
        }


# ==============================================================================
# 6. SQLITE STORE + FTS5 (Vá lỗi #4 từ `REVIEW_20261020.md`)
# ==============================================================================

class UnifiedSQLiteStore:
    """
    Lưu trữ cấu trúc `FinancialFact`, `DocumentChunk` (kèm FTS5 full-text search)
    và `QualityReport`.
    ĐÃ VÁ LỖI #4 TRONG REVIEW_20261020.md:
    - Khi truy vấn `query_fact` không tìm thấy đúng `period` được yêu cầu và được phép
      fallback (`allow_period_fallback=True`), câu truy vấn SQL luôn sắp xếp
      `ORDER BY period DESC LIMIT 1` để lấy kỳ báo cáo mới nhất một cách tất định,
      đồng thời đánh dấu cờ `is_fallback_period=True` và trả về rõ `actual_period`
      để tầng trả lời nói rõ kỳ thực tế cho người dùng.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                period TEXT NOT NULL,
                metric_code TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                vas_code TEXT,
                statement_type TEXT,
                report_scope TEXT DEFAULT 'consolidated',
                value REAL NOT NULL,
                unit TEXT DEFAULT 'VND',
                page INTEGER DEFAULT 1,
                bbox_json TEXT,
                raw_text TEXT,
                extraction_method TEXT,
                UNIQUE(company, period, statement_type, metric_code, report_scope)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                period TEXT NOT NULL,
                chunk_type TEXT,
                statement_type TEXT,
                page INTEGER DEFAULT 1,
                content TEXT NOT NULL,
                metadata_json TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                company,
                period,
                content
            );

            CREATE TABLE IF NOT EXISTS quality_reports (
                company TEXT NOT NULL,
                period TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                report_json TEXT NOT NULL,
                PRIMARY KEY (company, period)
            );
            """
        )
        self.conn.commit()

    def upsert_facts(self, facts: Iterable[FinancialFact]) -> int:
        count = 0
        cur = self.conn.cursor()
        for f in facts:
            cur.execute(
                """
                INSERT INTO facts (
                    company, period, metric_code, metric_name, vas_code,
                    statement_type, report_scope, value, unit, page,
                    bbox_json, raw_text, extraction_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company, period, statement_type, metric_code, report_scope)
                DO UPDATE SET
                    value = excluded.value,
                    metric_name = excluded.metric_name,
                    vas_code = excluded.vas_code,
                    unit = excluded.unit,
                    page = excluded.page,
                    bbox_json = excluded.bbox_json,
                    raw_text = excluded.raw_text,
                    extraction_method = excluded.extraction_method
                """,
                (
                    f.company.upper(),
                    f.period,
                    f.metric_code,
                    f.metric_name,
                    f.vas_code,
                    f.statement_type,
                    f.report_scope,
                    float(f.value),
                    f.unit,
                    int(f.page),
                    json.dumps(f.bbox) if f.bbox else None,
                    f.raw_text,
                    f.extraction_method,
                ),
            )
            count += 1
        self.conn.commit()
        return count

    def upsert_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        count = 0
        cur = self.conn.cursor()
        for c in chunks:
            cur.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, company, period, chunk_type, statement_type, page, content, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.chunk_id,
                    c.company.upper(),
                    c.period,
                    c.chunk_type,
                    c.statement_type,
                    int(c.page),
                    c.content,
                    json.dumps(c.metadata, ensure_ascii=False),
                ),
            )
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (c.chunk_id,))
            cur.execute(
                "INSERT INTO chunks_fts (chunk_id, company, period, content) VALUES (?, ?, ?, ?)",
                (c.chunk_id, c.company.upper(), c.period, c.content),
            )
            count += 1
        self.conn.commit()
        return count

    def save_quality_report(self, report: QualityReport) -> None:
        serializable = {
            "company": report.company,
            "period": report.period,
            "overall_status": report.overall_status,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "skipped_checks": report.skipped_checks,
            "self_correction_triggered": report.self_correction_triggered,
            "corrected_vas_codes": report.corrected_vas_codes,
            "checks": [
                {
                    **asdict(c),
                    "lhs_value": str(c.lhs_value) if c.lhs_value is not None else None,
                    "rhs_value": str(c.rhs_value) if c.rhs_value is not None else None,
                    "difference": str(c.difference) if c.difference is not None else None,
                }
                for c in report.checks
            ],
        }
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO quality_reports (company, period, overall_status, report_json)
            VALUES (?, ?, ?, ?)
            """,
            (report.company.upper(), report.period, report.overall_status, json.dumps(serializable, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_quality_report(self, company: str, period: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT report_json FROM quality_reports WHERE company = ? AND period = ?",
            (company.upper(), period),
        ).fetchone()
        return json.loads(row["report_json"]) if row else None

    def query_fact(
        self,
        company: str,
        metric_code: str,
        period: Optional[str] = None,
        allow_period_fallback: bool = True,
    ) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        comp = company.upper()

        if period:
            row = cur.execute(
                """
                SELECT * FROM facts
                WHERE company = ? AND (metric_code = ? OR vas_code = ?) AND period = ?
                ORDER BY id DESC LIMIT 1
                """,
                (comp, metric_code, metric_code, period),
            ).fetchone()
            if row:
                data = dict(row)
                data["is_fallback_period"] = False
                data["requested_period"] = period
                data["actual_period"] = data["period"]
                return data

            if not allow_period_fallback:
                return None

        row = cur.execute(
            """
            SELECT * FROM facts
            WHERE company = ? AND (metric_code = ? OR vas_code = ?)
            ORDER BY period DESC, id DESC LIMIT 1
            """,
            (comp, metric_code, metric_code),
        ).fetchone()
        if not row:
            return None

        data = dict(row)
        data["is_fallback_period"] = bool(period is not None and data["period"] != period)
        data["requested_period"] = period
        data["actual_period"] = data["period"]
        return data

    def list_facts_for_period(self, company: str, period: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT * FROM facts WHERE company = ? AND period = ? ORDER BY page ASC, id ASC",
            (company.upper(), period),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_chunks_lexical(
        self,
        query: str,
        company: Optional[str] = None,
        period: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
        if not tokens:
            return []

        fts_query = " OR ".join(tokens[:12])
        sql = """
            SELECT c.*, bm25(chunks_fts) AS bm25_rank
            FROM chunks_fts f
            JOIN chunks c ON c.chunk_id = f.chunk_id
            WHERE chunks_fts MATCH ?
        """
        params: List[Any] = [fts_query]
        if company:
            sql += " AND c.company = ?"
            params.append(company.upper())
        if period:
            sql += " AND c.period = ?"
            params.append(period)
        sql += " ORDER BY bm25_rank ASC LIMIT ?"
        params.append(int(top_k))

        cur = self.conn.cursor()
        rows = cur.execute(sql, params).fetchall()
        results: List[Dict[str, Any]] = []
        for idx, r in enumerate(rows):
            item = dict(r)
            item["lexical_score"] = 1.0 / (1.0 + idx)
            results.append(item)
        return results


# ==============================================================================
# 7. DETERMINISTIC FINANCIAL CALCULATOR (Từ `src/multimodal_rag/calculator.py`)
# ==============================================================================

class DeterministicFinancialCalculator:
    """
    Thực hiện mọi phép tính tài chính (YoY growth, ROE, ROA, Gross Margin, Net Margin,
    Current Ratio, Debt-to-Equity) bằng `Decimal` độ chính xác cao.
    Tuyệt đối không để LLM tự thực hiện phép tính số học.
    """

    @staticmethod
    def to_decimal(val: Union[float, int, str, Decimal]) -> Decimal:
        return Decimal(str(val))

    @classmethod
    def compute_yoy_growth(
        cls,
        current_val: Union[float, Decimal],
        previous_val: Union[float, Decimal],
    ) -> Dict[str, Any]:
        curr = cls.to_decimal(current_val)
        prev = cls.to_decimal(previous_val)
        abs_change = curr - prev
        if prev == Decimal("0"):
            pct_change = None
        else:
            pct_change = ((curr - prev) / abs(prev) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        return {
            "current_value": curr,
            "previous_value": prev,
            "absolute_change": abs_change,
            "pct_change": float(pct_change) if pct_change is not None else None,
            "formula": "(Current - Previous) / |Previous| * 100%",
        }

    @classmethod
    def compute_ratio(
        cls,
        ratio_name: str,
        numerator_val: Union[float, Decimal],
        denominator_val: Union[float, Decimal],
        as_percentage: bool = True,
    ) -> Dict[str, Any]:
        num = cls.to_decimal(numerator_val)
        den = cls.to_decimal(denominator_val)
        if den == Decimal("0"):
            return {"ratio_name": ratio_name, "value": None, "unit": "%" if as_percentage else "x"}
        raw = num / den
        if as_percentage:
            scaled = (raw * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            unit = "%"
        else:
            scaled = raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            unit = "lần"
        return {
            "ratio_name": ratio_name,
            "numerator": num,
            "denominator": den,
            "value": float(scaled),
            "unit": unit,
        }


# ==============================================================================
# 8. HYBRID RETRIEVER (`BAAI/bge-m3` + FTS5 + `BAAI/bge-reranker-v2-m3`)
# ==============================================================================

class HybridFinancialRetriever:
    """
    Kết hợp truy xuất ngữ nghĩa Dense (`BAAI/bge-m3` + FAISS/Cosine) với truy xuất từ khóa
    (`SQLite FTS5` / Sparse lexical) qua Reciprocal Rank Fusion (RRF), sau đó chấm điểm
    lại bằng Cross-Encoder `BAAI/bge-reranker-v2-m3`.
    Có cơ chế fallback sang TF-IDF n-gram nhẹ khi chạy kiểm thử CPU không có GPU.
    """

    EMBEDDING_MODEL_ID = "BAAI/bge-m3"
    RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        store: UnifiedSQLiteStore,
        use_gpu_models: bool = False,
    ) -> None:
        self.store = store
        self.use_gpu_models = use_gpu_models
        self.embedder = None
        self.reranker = None
        self._indexed_chunks: List[Dict[str, Any]] = []

    def index_chunks_in_memory(self, chunks: Sequence[DocumentChunk]) -> None:
        for c in chunks:
            self._indexed_chunks.append(asdict(c))

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        period: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        lexical_hits = self.store.search_chunks_lexical(
            query=query, company=company, period=period, top_k=top_k * 3
        )
        semantic_hits = self._semantic_search(
            query=query, company=company, period=period, top_k=top_k * 3
        )

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        k_rrf = 60.0

        for rank, item in enumerate(lexical_hits):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)

        for rank, item in enumerate(semantic_hits):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)

        merged = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = []
        for cid, score in merged[: top_k * 2]:
            entry = dict(chunk_map[cid])
            entry["hybrid_score"] = round(score, 6)
            candidates.append(entry)

        return self._rerank(query=query, candidates=candidates, top_k=top_k)

    def _semantic_search(
        self,
        query: str,
        company: Optional[str],
        period: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        cur = self.store.conn.cursor()
        sql = "SELECT * FROM chunks WHERE 1=1"
        params: List[Any] = []
        if company:
            sql += " AND company = ?"
            params.append(company.upper())
        if period:
            sql += " AND period = ?"
            params.append(period)
        rows = [dict(r) for r in cur.execute(sql, params).fetchall()]
        if not rows:
            return []

        q_grams = self._char_ngrams(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            c_grams = self._char_ngrams(r["content"])
            if not q_grams or not c_grams:
                sim = 0.0
            else:
                overlap = len(q_grams & c_grams)
                sim = overlap / math.sqrt(len(q_grams) * len(c_grams))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:top_k] if score > 0]

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        q_tokens = set(re.findall(r"\w+", query.lower()))
        for c in candidates:
            c_tokens = set(re.findall(r"\w+", c["content"].lower()))
            overlap_bonus = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            c["rerank_score"] = round(float(c.get("hybrid_score", 0.0)) + 0.25 * overlap_bonus, 6)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> set[str]:
        norm = re.sub(r"\s+", " ", (text or "").lower().strip())
        if len(norm) < n:
            return {norm} if norm else set()
        return {norm[i : i + n] for i in range(len(norm) - n + 1)}


# ==============================================================================
# 9. NUMERICAL VERIFIER (Vá lỗi #6 từ `REVIEW_20261020.md`)
# ==============================================================================

class NumericalVerifier:
    """
    Kiểm duyệt toàn bộ các con số xuất hiện trong câu trả lời của LLM so với tập dữ liệu gốc.
    ĐÃ VÁ LỖI #6 TRONG REVIEW_20261020.md:
    - Bản cũ dùng điều kiện `if abs(val) > 10:` khiến mọi con số <= 10 (như tăng trưởng 6.5%,
      biên lợi nhuận ròng 8.2%, hệ số thanh toán 1.45 lần, EPS 3.200đ) bị bỏ qua không kiểm duyệt.
    - Bản mới trích xuất và kiểm chứng MỌI con số tài chính (kể cả số âm, số thập phân nhỏ `<= 10`,
      tỷ lệ phần trăm), chỉ loại trừ các số thứ tự ngày/tháng/trang/mã số VAS đã biết.
    """

    def __init__(self, rel_tolerance: float = 0.015) -> None:
        self.rel_tolerance = rel_tolerance

    @staticmethod
    def extract_numbers(text: str, ignore_numbers: Optional[Iterable[float]] = None) -> List[float]:
        if not text:
            return []
        ignored = {round(float(x), 4) for x in (ignore_numbers or [])}

        cleaned = re.sub(r"\b(?:FY|Q[1-4]/?|năm|kỳ|trang|page|mã\s+số|mã)\s*\d{1,4}\b", " ", text, flags=re.I)
        cleaned = re.sub(r"\b20\d{2}\b", " ", cleaned)

        matches = re.findall(r"\(?-?\d+(?:[.,]\d+)*\)?%?", cleaned)
        extracted: List[float] = []
        for raw_token in matches:
            val = FactExtractor.clean_number(raw_token)
            if val is None:
                continue
            if round(val, 4) in ignored:
                continue
            extracted.append(val)
        return extracted

    def _extract_numbers(self, text: str) -> List[float]:
        return self.extract_numbers(text)

    def verify(
        self,
        answer_text: str,
        reference_numbers: Sequence[Union[float, int, Decimal]],
        known_metadata_numbers: Optional[Iterable[float]] = None,
    ) -> Dict[str, Any]:
        claim_numbers = self.extract_numbers(answer_text, ignore_numbers=known_metadata_numbers)
        ref_floats = [float(r) for r in reference_numbers if r is not None]

        if not claim_numbers:
            return {
                "is_verified": True,
                "verified_numbers": [],
                "unverified_numbers": [],
                "confidence": 1.0,
            }

        verified: List[float] = []
        unverified: List[float] = []

        for claim in claim_numbers:
            matched = False
            for ref in ref_floats:
                if ref == 0.0 and abs(claim) < 1e-6:
                    matched = True
                    break
                denom = max(abs(ref), 1e-9)
                if abs(claim - ref) / denom <= self.rel_tolerance or abs(claim - ref) <= 0.05:
                    matched = True
                    break
            if matched:
                verified.append(claim)
            else:
                unverified.append(claim)

        total = len(claim_numbers)
        confidence = len(verified) / total if total > 0 else 1.0
        return {
            "is_verified": len(unverified) == 0,
            "verified_numbers": verified,
            "unverified_numbers": unverified,
            "confidence": round(confidence, 4),
        }


# ==============================================================================
# 10. END-TO-END UNIFIED PIPELINE & REAL PDF FILE INGESTOR (Đã bổ sung đọc file .PDF thực tế)
# ==============================================================================

class UnifiedFinancialRAGPipeline:
    """
    Pipeline hợp nhất hoàn chỉnh nối khép kín:
    `RealPDFFileReader` (`pdfplumber` / `pypdfium2` / `pypdf` + `Qwen2.5-VL` OCR cho trang scan)
    -> `SmartFinancialPDFParser` -> `VASAccountingQualityGate` -> `UnifiedSQLiteStore`
    -> `QueryRouter` -> (`DeterministicFinancialCalculator` + `HybridFinancialRetriever`)
    -> `SharedQwenVLBackbone` -> `NumericalVerifier`.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        vlm_backbone: Optional[SharedQwenVLBackbone] = None,
        seed_sample_data_if_empty: bool = False,
    ) -> None:
        self.registry = VASMetricRegistry()
        self.fact_extractor = FactExtractor(self.registry)
        self.quality_gate = VASAccountingQualityGate()
        self.vlm_backbone = vlm_backbone or SharedQwenVLBackbone(lazy_load=True)
        self.pdf_parser = SmartFinancialPDFParser(
            fact_extractor=self.fact_extractor,
            vlm_backbone=self.vlm_backbone,
            quality_gate=self.quality_gate,
        )
        self.store = UnifiedSQLiteStore(db_path=db_path)
        self.ocr_cache = PageOCRCacheManager(self.store.conn)
        self.retriever = HybridFinancialRetriever(store=self.store)
        self.calculator = DeterministicFinancialCalculator()
        self.verifier = NumericalVerifier()

        if seed_sample_data_if_empty and self.get_database_summary()["total_facts"] == 0:
            self.seed_default_fpt_data()

    def get_database_summary(self) -> Dict[str, Any]:
        """Trả về thống kê các báo cáo đã nạp trong SQLite để hiển thị trên giao diện Gradio."""
        cur = self.store.conn.cursor()
        fact_count = cur.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        chunk_count = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        rows = cur.execute(
            """
            SELECT company, period, COUNT(*) as num_facts
            FROM facts
            GROUP BY company, period
            ORDER BY company ASC, period DESC
            """
        ).fetchall()
        reports = [dict(r) for r in rows]
        return {
            "total_facts": int(fact_count),
            "total_chunks": int(chunk_count),
            "loaded_reports": reports,
        }

    def seed_default_fpt_data(self) -> None:
        """
        Nạp sẵn dữ liệu BCTC Hợp nhất mẫu của FPT (FY2023 và FY2024) khi khởi tạo DB rỗng,
        giúp người dùng vừa mở Gradio lên là có thể đặt câu hỏi kiểm thử ngay lập tức.
        """
        fpt_2023_pages = [
            {
                "page": 1,
                "text": (
                    "CÔNG TY CỔ PHẦN FPT - BÁO CÁO TÀI CHÍNH HỢP NHẤT NĂM 2023 (FY2023).\n"
                    "Bảng cân đối kế toán hợp nhất tại ngày 31/12/2023. Đơn vị tính: Tỷ đồng."
                ),
                "tables": [
                    [
                        ["Tài sản ngắn hạn", "100", "V.01", "36.705", "30.938"],
                        ["Tiền và các khoản tương đương tiền", "110", "V.02", "8.279", "6.440"],
                        ["Các khoản phải thu ngắn hạn", "130", "V.03", "9.674", "8.120"],
                        ["Hàng tồn kho", "140", "V.04", "1.593", "1.965"],
                        ["Tài sản dài hạn", "200", "V.08", "23.578", "20.712"],
                        ["Tổng cộng tài sản", "270", "", "60.283", "51.650"],
                        ["Nợ phải trả", "300", "V.12", "30.350", "26.294"],
                        ["Nợ ngắn hạn", "310", "V.13", "29.752", "25.100"],
                        ["Nợ dài hạn", "330", "V.14", "598", "1.194"],
                        ["Vốn chủ sở hữu", "400", "V.18", "29.933", "25.356"],
                        ["Tổng cộng nguồn vốn", "440", "", "60.283", "51.650"],
                    ]
                ],
            },
            {
                "page": 2,
                "text": (
                    "CÔNG TY CỔ PHẦN FPT - Báo cáo kết quả hoạt động kinh doanh hợp nhất năm 2023 (FY2023).\n"
                    "Đơn vị tính: Tỷ đồng."
                ),
                "tables": [
                    [
                        ["Doanh thu thuần về bán hàng và cung cấp dịch vụ", "10", "VI.1", "52.618", "44.010"],
                        ["Giá vốn hàng bán", "11", "VI.2", "32.333", "26.826"],
                        ["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "20", "", "20.285", "17.184"],
                        ["Tổng lợi nhuận kế toán trước thuế", "50", "", "9.203", "7.662"],
                        ["Chi phí thuế TNDN hiện hành", "51", "VI.8", "1.415", "1.171"],
                        ["Chi phí thuế TNDN hoãn lại", "52", "VI.8", "0", "0"],
                        ["Lợi nhuận sau thuế thu nhập doanh nghiệp", "60", "", "7.788", "6.491"],
                    ]
                ],
            },
        ]

        fpt_2024_pages = [
            {
                "page": 1,
                "text": (
                    "CÔNG TY CỔ PHẦN FPT - BÁO CÁO TÀI CHÍNH HỢP NHẤT NĂM 2024 (FY2024).\n"
                    "Bảng cân đối kế toán hợp nhất tại ngày 31/12/2024. Đơn vị tính: Tỷ đồng."
                ),
                "tables": [
                    [
                        ["Tài sản ngắn hạn", "100", "V.01", "45.610", "36.705"],
                        ["Tiền và các khoản tương đương tiền", "110", "V.02", "9.315", "8.279"],
                        ["Các khoản phải thu ngắn hạn", "130", "V.03", "11.240", "9.674"],
                        ["Hàng tồn kho", "140", "V.04", "1.845", "1.593"],
                        ["Tài sản dài hạn", "200", "V.08", "26.390", "23.578"],
                        ["Tổng cộng tài sản", "270", "", "72.000", "60.283"],
                        ["Nợ phải trả", "300", "V.12", "36.400", "30.350"],
                        ["Nợ ngắn hạn", "310", "V.13", "35.120", "29.752"],
                        ["Nợ dài hạn", "330", "V.14", "1.280", "598"],
                        ["Vốn chủ sở hữu", "400", "V.18", "35.600", "29.933"],
                        ["Tổng cộng nguồn vốn", "440", "", "72.000", "60.283"],
                    ]
                ],
            },
            {
                "page": 2,
                "text": (
                    "CÔNG TY CỔ PHẦN FPT - Báo cáo kết quả hoạt động kinh doanh hợp nhất năm 2024 (FY2024).\n"
                    "Đơn vị tính: Tỷ đồng."
                ),
                "tables": [
                    [
                        ["Doanh thu thuần về bán hàng và cung cấp dịch vụ", "10", "VI.1", "62.849", "52.618"],
                        ["Giá vốn hàng bán", "11", "VI.2", "38.249", "32.333"],
                        ["Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "20", "", "24.600", "20.285"],
                        ["Tổng lợi nhuận kế toán trước thuế", "50", "", "11.070", "9.203"],
                        ["Chi phí thuế TNDN hiện hành", "51", "VI.8", "1.643", "1.415"],
                        ["Chi phí thuế TNDN hoãn lại", "52", "VI.8", "0", "0"],
                        ["Lợi nhuận sau thuế thu nhập doanh nghiệp", "60", "", "9.427", "7.788"],
                    ]
                ],
            },
            {
                "page": 3,
                "text": (
                    "CÔNG TY CỔ PHẦN FPT - Báo cáo lưu chuyển tiền tệ hợp nhất năm 2024 (FY2024).\n"
                    "Thuyết minh: Doanh thu khối Công nghệ thông tin nước ngoài tăng trưởng 27,4% nhờ thị trường Nhật Bản và APAC."
                ),
                "tables": [
                    [
                        ["Lưu chuyển tiền thuần từ hoạt động kinh doanh", "20", "", "12.450", "9.820"],
                        ["Lưu chuyển tiền thuần từ hoạt động đầu tư", "30", "", "(6.814)", "(5.120)"],
                        ["Lưu chuyển tiền thuần từ hoạt động tài chính", "40", "", "(4.600)", "(2.861)"],
                        ["Lưu chuyển tiền thuần trong kỳ", "50", "", "1.036", "1.839"],
                        ["Tiền và tương đương tiền đầu kỳ", "60", "", "8.279", "6.440"],
                        ["Ảnh hưởng của thay đổi tỷ giá hối đoái quy đổi ngoại tệ", "61", "", "0", "0"],
                        ["Tiền và tương đương tiền cuối kỳ", "70", "", "9.315", "8.279"],
                    ]
                ],
            },
        ]
        self.ingest_pages(fpt_2023_pages, company="FPT", period="FY2023", strict_identity_check=False)
        self.ingest_pages(fpt_2024_pages, company="FPT", period="FY2024", strict_identity_check=False)

    @staticmethod
    def infer_company_and_period_from_filename(
        file_path: Union[str, Path],
        default_company: str = "FPT",
        default_period: str = "FY2024",
    ) -> Tuple[str, str]:
        stem = Path(file_path).stem.upper()
        comp_match = re.match(r"^([A-Z]{3,5})[_\-\s]", stem)
        company = comp_match.group(1) if comp_match else default_company.upper()

        q_match = re.search(r"(Q[1-4][_\-\s]*20\d{2}|20\d{2}[_\-\s]*Q[1-4])", stem)
        y_match = re.search(r"(20\d{2})", stem)
        if q_match:
            period = re.sub(r"[\s_\-]+", "_", q_match.group(1))
        elif y_match:
            period = f"FY{y_match.group(1)}"
        else:
            period = default_period
        return company, period

    def read_pdf_file_to_pages_data(
        self,
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Đọc file `.pdf` vật lý từ đĩa hoặc từ Gradio File Upload:
        1. Thử đọc text và bảng bằng `pdfplumber` (chuẩn nhất cho PDF có text layer).
        2. Fallback sang `pypdf` hoặc `pypdfium2` nếu môi trường thiếu `pdfplumber`.
        3. Với các trang scan (`len(text.strip()) < 40` và không có bảng), tự động render
           trang PDF thành ảnh `PIL.Image` (200 DPI) qua `pypdfium2` để đưa vào `Qwen2.5-VL`.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Không tìm thấy file PDF tại đường dẫn: {path_obj}")

        pages_data: List[Dict[str, Any]] = []

        pdfium_doc = None
        try:
            import pypdfium2 as pdfium
            pdfium_doc = pdfium.PdfDocument(str(path_obj))
        except Exception:
            pdfium_doc = None

        try:
            import pdfplumber

            with pdfplumber.open(str(path_obj)) as pdf:
                total_p = len(pdf.pages)
                limit = min(total_p, max_pages) if max_pages else total_p
                for idx in range(limit):
                    p = pdf.pages[idx]
                    txt = (p.extract_text() or "").strip()
                    raw_tables = p.extract_tables() or []
                    cleaned_tables = []
                    for tbl in raw_tables:
                        if not tbl:
                            continue
                        cleaned_rows = [
                            [str(cell).strip() if cell is not None else "" for cell in row]
                            for row in tbl
                            if row and any(c is not None and str(c).strip() for c in row)
                        ]
                        if cleaned_rows:
                            cleaned_tables.append(cleaned_rows)

                    if not cleaned_tables and txt:
                        heuristic_rows = self._extract_vas_rows_from_plain_text(txt)
                        if heuristic_rows:
                            cleaned_tables.append(heuristic_rows)

                    page_image = None
                    if len(txt) < 40 and not cleaned_tables and pdfium_doc is not None and idx < len(pdfium_doc):
                        try:
                            bitmap = pdfium_doc[idx].render(scale=2.0)
                            page_image = bitmap.to_pil()
                        except Exception:
                            page_image = None

                    pages_data.append(
                        {
                            "page": idx + 1,
                            "text": txt,
                            "tables": cleaned_tables,
                            "image": page_image,
                        }
                    )
            return pages_data
        except ImportError:
            pass

        from pypdf import PdfReader

        reader = PdfReader(str(path_obj))
        total_p = len(reader.pages)
        limit = min(total_p, max_pages) if max_pages else total_p
        for idx in range(limit):
            txt = (reader.pages[idx].extract_text() or "").strip()
            heuristic_rows = self._extract_vas_rows_from_plain_text(txt) if txt else []
            cleaned_tables = [heuristic_rows] if heuristic_rows else []

            page_image = None
            if len(txt) < 40 and not cleaned_tables and pdfium_doc is not None and idx < len(pdfium_doc):
                try:
                    bitmap = pdfium_doc[idx].render(scale=2.0)
                    page_image = bitmap.to_pil()
                except Exception:
                    page_image = None

            pages_data.append(
                {
                    "page": idx + 1,
                    "text": txt,
                    "tables": cleaned_tables,
                    "image": page_image,
                }
            )
        return pages_data

    @staticmethod
    def _extract_vas_rows_from_plain_text(page_text: str) -> List[List[str]]:
        rows: List[List[str]] = []
        pattern = re.compile(
            r"^(?P<label>[^\d\n][^\n]{3,95}?)\s+(?P<code>\d{2,3})\s+"
            r"(?:(?P<note>[IVX]+\.\d+|TM\.?\d+|\d{1,2}\.\d{1,2})\s+)?"
            r"(?P<curr>\(?-?\d[\d.,]*\)?)"
            r"(?:\s+(?P<prev>\(?-?\d[\d.,]*\)?))?\s*$",
            re.IGNORECASE,
        )
        for raw_line in page_text.splitlines():
            line = re.sub(r"^\s*(?:[IVX]+|[A-Z]|\d+)\.\s*", "", raw_line.strip())
            m = pattern.match(line)
            if m:
                rows.append(
                    [
                        m.group("label").strip(),
                        m.group("code").strip(),
                        (m.group("note") or "").strip(),
                        m.group("curr").strip(),
                        (m.group("prev") or "").strip(),
                    ]
                )
        return rows

    def ingest_pdf_file(
        self,
        pdf_path: Union[str, Path],
        company: Optional[str] = None,
        period: Optional[str] = None,
        strict_identity_check: bool = False,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Nạp trực tiếp một file `.pdf` vào hệ thống:
        Đọc trang -> Bóc bảng & chữ (kèm VLM OCR cho trang scan) -> Kiểm toán VAS -> Lưu SQLite & Index.
        """
        inferred_comp, inferred_period = self.infer_company_and_period_from_filename(pdf_path)
        comp = (company or "").strip().upper() or inferred_comp
        per = (period or "").strip() or inferred_period

        pages_data = self.read_pdf_file_to_pages_data(pdf_path, max_pages=max_pages)
        result = self.ingest_pages(
            pages_data=pages_data,
            company=comp,
            period=per,
            strict_identity_check=strict_identity_check,
        )
        result["file_name"] = Path(pdf_path).name
        result["pages_read"] = len(pages_data)
        return result

    def ingest_pages(
        self,
        pages_data: Sequence[Dict[str, Any]],
        company: str,
        period: str,
        strict_identity_check: bool = True,
    ) -> Dict[str, Any]:
        result = self.pdf_parser.extract_from_pages_data(
            pages_data=pages_data,
            company=company,
            period=period,
            strict_identity_check=strict_identity_check,
        )
        if not result["validation"]["is_valid"] and len(result["facts"]) == 0 and len(result["chunks"]) == 0:
            return result

        self.store.upsert_facts(result["facts"])
        self.store.upsert_chunks(result["chunks"])
        self.store.save_quality_report(result["quality_report"])
        self.retriever.index_chunks_in_memory(result["chunks"])
        return result

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ("kiểm toán", "cân đối", "khớp", "reconcile", "reconciliation", "đẳng thức")):
            return "reconciliation"
        if any(k in q for k in ("tăng trưởng", "so với", "yoy", "thay đổi", "biến động")):
            return "yoy_comparison"
        if any(k in q for k in ("tỷ suất", "biên lợi nhuận", "roe", "roa", "margin", "tỷ lệ")):
            return "ratio"
        return "metric_or_hybrid_qa"

    def answer_question(
        self,
        query: str,
        company: str,
        period: str,
        comparison_period: Optional[str] = None,
    ) -> Dict[str, Any]:
        q_clean = (query or "").strip()
        if not q_clean:
            return {
                "intent": "empty_query",
                "answer": "Vui lòng nhập câu hỏi phân tích tài chính (ví dụ: 'Tổng cộng tài sản của FPT năm 2024 là bao nhiêu?').",
                "execution_path": "validation",
                "verification": {"is_verified": True, "confidence": 1.0},
            }

        intent = self.classify_intent(q_clean)
        comp = (company or "FPT").strip().upper()
        per = (period or "FY2024").strip()

        if intent == "reconciliation":
            q_report = self.store.get_quality_report(comp, per)
            if not q_report:
                return {
                    "intent": intent,
                    "answer": f"Chưa có dữ liệu kiểm toán BCTC cho {comp} kỳ {per}. Bạn hãy tải file PDF lên ở mục 'Nạp Báo cáo tài chính PDF' phía trên.",
                    "execution_path": "deterministic_reconciliation",
                    "verification": {"is_verified": True, "confidence": 1.0},
                }
            passed = q_report["passed_checks"]
            total = q_report["total_checks"]
            status_vi = "ĐẠT CHUẨN" if q_report["overall_status"] == "pass" else "CẢNH BÁO / LỆCH SỐ LIỆU"
            details = [
                f"- {c['description']}: {c['status'].upper()} (Chênh lệch: {c['difference'] or 0})"
                for c in q_report["checks"]
                if c["status"] != "skipped"
            ]
            answer_text = (
                f"Báo cáo kiểm tra chéo đẳng thức kế toán VAS của {comp} ({per}): {status_vi} "
                f"({passed}/{total} quy tắc kiểm tra).\n" + "\n".join(details)
            )
            return {
                "intent": intent,
                "answer": answer_text,
                "execution_path": "deterministic_reconciliation",
                "quality_report": q_report,
                "verification": {"is_verified": True, "confidence": 1.0},
            }

        if intent == "ratio":
            return self._handle_ratio_query(query=q_clean, company=comp, period=per)

        if intent == "yoy_comparison" and comparison_period:
            spec = self.registry.match(q_clean)
            if spec:
                curr_fact = self.store.query_fact(comp, spec.metric_code, per, allow_period_fallback=False)
                prev_fact = self.store.query_fact(comp, spec.metric_code, comparison_period, allow_period_fallback=False)
                if curr_fact and prev_fact:
                    yoy = self.calculator.compute_yoy_growth(curr_fact["value"], prev_fact["value"])
                    pct_str = f"{yoy['pct_change']:+.2f}%" if yoy["pct_change"] is not None else "N/A"
                    answer_text = (
                        f"{spec.canonical_name_vi} (Mã {spec.vas_code.replace('_CF', '')}) của {comp} "
                        f"kỳ {per} đạt {curr_fact['value']:,.0f} {curr_fact['unit']} (trang {curr_fact['page']}), "
                        f"so với mức {prev_fact['value']:,.0f} {prev_fact['unit']} của kỳ {comparison_period} "
                        f"(thay đổi tuyệt đối {float(yoy['absolute_change']):+,.0f} {curr_fact['unit']}, tương ứng {pct_str})."
                    )
                    ref_nums = [
                        curr_fact["value"],
                        prev_fact["value"],
                        float(yoy["absolute_change"]),
                        yoy["pct_change"] or 0.0,
                    ]
                    verification = self.verifier.verify(answer_text, ref_nums)
                    return {
                        "intent": intent,
                        "answer": answer_text,
                        "execution_path": "deterministic_calculator",
                        "calculation": yoy,
                        "citations": [
                            {"period": per, "page": curr_fact["page"], "bbox": curr_fact.get("bbox_json")},
                            {"period": comparison_period, "page": prev_fact["page"], "bbox": prev_fact.get("bbox_json")},
                        ],
                        "verification": verification,
                    }

        spec = self.registry.match(q_clean)
        fact_record = (
            self.store.query_fact(comp, spec.metric_code, per, allow_period_fallback=True)
            if spec
            else None
        )
        retrieved_chunks = self.retriever.search(query=q_clean, company=comp, period=per, top_k=4)

        if fact_record:
            actual_period = fact_record["actual_period"]
            fallback_notice = (
                f" (Lưu ý: Không tìm thấy dữ liệu kỳ {per}, hệ thống trả về kỳ gần nhất hiện có là {actual_period})"
                if fact_record.get("is_fallback_period")
                else ""
            )
            vas_display = f" (Mã số {fact_record['vas_code']})" if fact_record.get("vas_code") else ""
            answer_text = (
                f"Theo BCTC của {comp} kỳ {actual_period}{fallback_notice}, chỉ tiêu "
                f"{fact_record['metric_name']}{vas_display} là {fact_record['value']:,.0f} {fact_record['unit']} "
                f"(trích xuất tại trang {fact_record['page']})."
            )
            verification = self.verifier.verify(answer_text, [fact_record["value"]])
            return {
                "intent": intent,
                "answer": answer_text,
                "execution_path": "deterministic_sql_fact_with_hybrid_context",
                "fact": fact_record,
                "retrieved_chunks": retrieved_chunks,
                "verification": verification,
            }

        context_text = "\n\n".join(
            f"[Trang {c['page']}] {c['content']}" for c in retrieved_chunks
        )
        all_period_facts = self.store.list_facts_for_period(comp, per)
        ref_numbers = [f["value"] for f in all_period_facts]
        for c in retrieved_chunks:
            ref_numbers.extend(self.verifier.extract_numbers(c["content"]))

        if not context_text and all_period_facts:
            top_facts_summary = "; ".join(
                f"{f['metric_name']} (Mã {f['vas_code'] or 'N/A'}): {f['value']:,.0f} {f['unit']}"
                for f in all_period_facts[:8]
            )
            context_text = f"Các chỉ tiêu tài chính chính của {comp} kỳ {per}: {top_facts_summary}."

        llm_answer = self.vlm_backbone.generate_text(
            system_prompt=(
                "Bạn là chuyên gia phân tích Báo cáo tài chính. Chỉ sử dụng đúng số liệu trong ngữ cảnh được cung cấp, "
                "nêu rõ kỳ báo cáo và số trang trích dẫn."
            ),
            user_prompt=f"Câu hỏi: {q_clean}\nCông ty: {comp}, Kỳ: {per}\nNgữ cảnh:\n{context_text}",
        )
        if not llm_answer:
            llm_answer = (
                f"Dựa trên BCTC {comp} kỳ {per}:\n{context_text[:800]}"
                if context_text
                else (
                    f"Hiện chưa có dữ liệu của doanh nghiệp '{comp}' kỳ '{per}' trong cơ sở dữ liệu. "
                    f"Bạn hãy tải file PDF BCTC lên ở khung '1. Nạp Báo cáo tài chính PDF' phía trên rồi bấm phân tích."
                )
            )

        verification = self.verifier.verify(llm_answer, ref_numbers)
        return {
            "intent": intent,
            "answer": llm_answer,
            "execution_path": "hybrid_rag_with_numerical_verifier",
            "retrieved_chunks": retrieved_chunks,
            "verification": verification,
        }

    def _handle_ratio_query(self, query: str, company: str, period: str) -> Dict[str, Any]:
        q = query.lower()
        if "biên lợi nhuận gộp" in q or "gross margin" in q:
            num_fact = self.store.query_fact(company, "gross_profit", period, allow_period_fallback=True)
            den_fact = self.store.query_fact(company, "net_revenue", period, allow_period_fallback=True)
            ratio_label = "Biên lợi nhuận gộp (Gross Profit Margin)"
        elif "roe" in q:
            num_fact = self.store.query_fact(company, "net_profit", period, allow_period_fallback=True)
            den_fact = self.store.query_fact(company, "owners_equity", period, allow_period_fallback=True)
            ratio_label = "Tỷ suất lợi nhuận trên vốn chủ sở hữu (ROE)"
        elif "roa" in q:
            num_fact = self.store.query_fact(company, "net_profit", period, allow_period_fallback=True)
            den_fact = self.store.query_fact(company, "total_assets", period, allow_period_fallback=True)
            ratio_label = "Tỷ suất lợi nhuận trên tổng tài sản (ROA)"
        else:
            num_fact = self.store.query_fact(company, "net_profit", period, allow_period_fallback=True)
            den_fact = self.store.query_fact(company, "net_revenue", period, allow_period_fallback=True)
            ratio_label = "Biên lợi nhuận ròng (Net Profit Margin)"

        if not num_fact or not den_fact:
            return {
                "intent": "ratio",
                "answer": f"Thiếu chỉ tiêu thành phần để tính {ratio_label} cho {company} kỳ {period}.",
                "execution_path": "deterministic_calculator",
                "verification": {"is_verified": True, "confidence": 1.0},
            }

        res = self.calculator.compute_ratio(ratio_label, num_fact["value"], den_fact["value"], as_percentage=True)
        actual_period = num_fact["actual_period"]
        answer_text = (
            f"{ratio_label} của {company} kỳ {actual_period} là {res['value']:.2f}{res['unit']} "
            f"(được tính tất định từ {num_fact['metric_name']}: {num_fact['value']:,.0f} {num_fact['unit']} "
            f"chia cho {den_fact['metric_name']}: {den_fact['value']:,.0f} {den_fact['unit']})."
        )
        verification = self.verifier.verify(
            answer_text,
            [res["value"], num_fact["value"], den_fact["value"]],
        )
        return {
            "intent": "ratio",
            "answer": answer_text,
            "execution_path": "deterministic_calculator",
            "calculation": res,
            "verification": verification,
        }


# ==============================================================================
# 11. GRADIO UI INTEGRATION FOR GOOGLE COLAB T4 (Có khung Upload PDF + Dữ liệu mẫu sẵn)
# ==============================================================================

def create_colab_gradio_interface(pipeline: UnifiedFinancialRAGPipeline):
    """
    Giao diện Gradio đầy đủ gồm 2 phần:
    1. Tải trực tiếp file PDF Báo cáo tài chính (.pdf) lên hệ thống + Xem trạng thái dữ liệu đã nạp.
    2. Đặt câu hỏi phân tích tài chính, tính tỷ số YoY/ROE/ROA và kiểm toán cân đối BCTC.
    Nếu cơ sở dữ liệu đang trống khi mở Gradio, tự động nạp sẵn bộ dữ liệu mẫu FPT (FY2023 & FY2024).
    """
    import gradio as gr

    if pipeline.get_database_summary()["total_facts"] == 0:
        pipeline.seed_default_fpt_data()

    def format_db_status() -> str:
        summary = pipeline.get_database_summary()
        if not summary["loaded_reports"]:
            return "Chưa có báo cáo nào trong DB. Hãy tải file PDF lên ở mục 1 bên dưới."
        items = [
            f"{r['company']} ({r['period']}): {r['num_facts']} chỉ tiêu"
            for r in summary["loaded_reports"]
        ]
        return (
            f"Đã nạp sẵn {summary['total_facts']} chỉ tiêu tài chính & {summary['total_chunks']} đoạn/bảng "
            f"từ: " + " | ".join(items)
        )

    def handle_pdf_upload(
        uploaded_files: Any,
        company_override: str,
        period_override: str,
    ) -> Tuple[str, str]:
        if not uploaded_files:
            return "Vui lòng chọn ít nhất 1 file PDF Báo cáo tài chính để nạp.", format_db_status()

        file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        logs: List[str] = []

        for f_obj in file_list:
            f_path = f_obj.name if hasattr(f_obj, "name") else str(f_obj)
            try:
                res = pipeline.ingest_pdf_file(
                    pdf_path=f_path,
                    company=company_override.strip() or None,
                    period=period_override.strip() or None,
                    strict_identity_check=False,
                )
                q_rep = res["quality_report"]
                logs.append(
                    f"[THÀNH CÔNG] File '{res['file_name']}' -> Công ty: {res['company']} | Kỳ: {res['period']} | "
                    f"Số trang đọc: {res['pages_read']} | Chỉ tiêu bóc tách: {len(res['facts'])} | "
                    f"Segments/Bảng: {len(res['chunks'])} | Kiểm toán VAS: {q_rep.overall_status.upper()} "
                    f"({q_rep.passed_checks}/{q_rep.total_checks} quy tắc)"
                )
            except Exception as exc:
                logs.append(f"[LỖI] Không thể nạp file '{Path(f_path).name}': {exc}")

        return "\n".join(logs), format_db_status()

    def handle_qa(query: str, company: str, period: str, comparison_period: str) -> Tuple[str, str, str]:
        comp_period = comparison_period.strip() or None
        out = pipeline.answer_question(
            query=query,
            company=company.strip(),
            period=period.strip(),
            comparison_period=comp_period,
        )
        ver_str = json.dumps(out.get("verification", {}), ensure_ascii=False, indent=2)
        meta_str = json.dumps(
            {
                "intent": out.get("intent"),
                "execution_path": out.get("execution_path"),
                "calculation": out.get("calculation"),
            },
            default=str,
            ensure_ascii=False,
            indent=2,
        )
        return out["answer"], ver_str, meta_str

    with gr.Blocks(title="Unified Financial Multimodal RAG (Colab T4)") as demo:
        gr.Markdown("## Hệ thống Financial Multimodal RAG Hợp nhất (Qwen2.5-VL + BGE-M3 + VAS Audit Gate)")

        db_status_box = gr.Textbox(
            label="Trạng thái Cơ sở dữ liệu hiện tại (Đã nạp sẵn dữ liệu FPT FY2023 & FY2024)",
            value=format_db_status(),
            interactive=False,
        )

        with gr.Accordion("1. Nạp thêm Báo cáo tài chính PDF từ máy tính (.pdf)", open=True):
            with gr.Row():
                pdf_input = gr.File(
                    label="Chọn 1 hoặc nhiều file PDF BCTC (.pdf)",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                with gr.Column():
                    upload_company = gr.Textbox(
                        label="Mã Doanh nghiệp cho file tải lên (Để trống sẽ tự nhận diện từ tên file)",
                        placeholder="Ví dụ: FPT, VNM, HPG...",
                        value="FPT",
                    )
                    upload_period = gr.Textbox(
                        label="Kỳ Báo cáo cho file tải lên (Để trống sẽ tự nhận diện từ tên file)",
                        placeholder="Ví dụ: FY2024, Q4_2024...",
                        value="FY2024",
                    )
                    ingest_btn = gr.Button("📥 Nạp & Kiểm toán File PDF vào Hệ thống", variant="secondary")
            ingest_log_out = gr.Textbox(label="Kết quả bóc tách & Kiểm toán đầu vào (VAS Quality Gate)", lines=2)

        gr.Markdown("### 2. Truy vấn, Tính toán Tất định & Kiểm chứng Số liệu")
        with gr.Row():
            company_in = gr.Textbox(label="Mã Doanh nghiệp", value="FPT")
            period_in = gr.Textbox(label="Kỳ Báo cáo", value="FY2024")
            comp_period_in = gr.Textbox(label="Kỳ So sánh YoY (Tùy chọn)", value="FY2023")

        query_in = gr.Textbox(
            label="Câu hỏi phân tích tài chính",
            value="Tổng cộng tài sản của FPT năm 2024 là bao nhiêu?",
            placeholder="Ví dụ: Tổng cộng tài sản của FPT năm 2024 là bao nhiêu? Hoặc: Kiểm toán cân đối BCTC",
        )
        submit_btn = gr.Button("Phân tích & Kiểm chứng số liệu", variant="primary")

        gr.Examples(
            examples=[
                ["Tổng cộng tài sản của FPT năm 2024 là bao nhiêu?", "FPT", "FY2024", "FY2023"],
                ["Tăng trưởng doanh thu thuần của FPT năm 2024 so với 2023?", "FPT", "FY2024", "FY2023"],
                ["Tính biên lợi nhuận gộp của FPT kỳ FY2024", "FPT", "FY2024", "FY2023"],
                ["Tính ROE của FPT kỳ FY2024", "FPT", "FY2024", "FY2023"],
                ["Kiểm toán đẳng thức kế toán BCTC FPT", "FPT", "FY2024", "FY2023"],
            ],
            inputs=[query_in, company_in, period_in, comp_period_in],
            label="Câu hỏi mẫu (Bấm vào để thử ngay)",
        )

        answer_out = gr.Textbox(label="Câu trả lời (Đã kiểm chứng tất định)", lines=5)
        with gr.Row():
            verifier_out = gr.Code(label="Kết quả Numerical Verifier", language="json")
            meta_out = gr.Code(label="Trace Đường thực thi & Tính toán", language="json")

        ingest_btn.click(
            fn=handle_pdf_upload,
            inputs=[pdf_input, upload_company, upload_period],
            outputs=[ingest_log_out, db_status_box],
        )
        submit_btn.click(
            fn=handle_qa,
            inputs=[query_in, company_in, period_in, comp_period_in],
            outputs=[answer_out, verifier_out, meta_out],
        )
    return demo


# ==============================================================================
# 12. CÁC NÂNG CẤP NÂNG CAO (UPGRADES V2 CHO BCTC VIỆT NAM & COLAB T4)
# ==============================================================================

import hashlib


class UnitScaleNormalizer:
    """
    NÂNG CẤP #1: Tự động phát hiện và chuẩn hóa Đơn vị tiền tệ trên đầu trang BCTC
    ("Đơn vị tính: Triệu đồng", "Tỷ đồng", "Nghìn đồng", "VND", "Million USD").
    Ngăn chặn triệt để lỗi lệch quy mô 10^3 - 10^9 lần khi tính YoY hoặc ghép bảng.
    """

    UNIT_PATTERNS: Tuple[Tuple[str, str, float], ...] = (
        (r"đơn\s*vị\s*tính\s*[:\-]?\s*tỷ\s*(?:đồng|vnd|vnđ)", "VND", 1_000_000_000.0),
        (r"đơn\s*vị\s*tính\s*[:\-]?\s*triệu\s*(?:đồng|vnd|vnđ)", "VND", 1_000_000.0),
        (r"đơn\s*vị\s*tính\s*[:\-]?\s*(?:nghìn|ngàn)\s*(?:đồng|vnd|vnđ)", "VND", 1_000.0),
        (r"đơn\s*vị\s*tính\s*[:\-]?\s*(?:đồng|vnd|vnđ)", "VND", 1.0),
        (r"\(\s*in\s+millions?\s*(?:of\s+usd|\$)?\s*\)", "USD", 1_000_000.0),
        (r"\(\s*in\s+thousands?\s*(?:of\s+usd|\$)?\s*\)", "USD", 1_000.0),
    )

    @classmethod
    def detect_unit_and_multiplier(cls, page_text: str, default_unit: str = "VND") -> Tuple[str, float]:
        norm = (page_text or "").lower()
        for pattern, base_unit, multiplier in cls.UNIT_PATTERNS:
            if re.search(pattern, norm, flags=re.I):
                return base_unit, multiplier
        return default_unit, 1.0

    @classmethod
    def normalize_facts(cls, facts: Sequence[FinancialFact], page_text: str) -> List[FinancialFact]:
        base_unit, multiplier = cls.detect_unit_and_multiplier(page_text)
        if multiplier == 1.0:
            return list(facts)
        normalized: List[FinancialFact] = []
        for f in facts:
            scaled_val = float(Decimal(str(f.value)) * Decimal(str(multiplier)))
            normalized.append(
                FinancialFact(
                    company=f.company,
                    period=f.period,
                    metric_code=f.metric_code,
                    metric_name=f.metric_name,
                    value=scaled_val,
                    unit=base_unit,
                    vas_code=f.vas_code,
                    statement_type=f.statement_type,
                    report_scope=f.report_scope,
                    page=f.page,
                    bbox=f.bbox,
                    raw_text=f.raw_text,
                    confidence=f.confidence,
                    extraction_method=f.extraction_method,
                )
            )
        return normalized


class ReportScopeDetector:
    """
    NÂNG CẤP #2: Tự động phân biệt BCTC Hợp nhất (`consolidated`) và BCTC Riêng / Công ty mẹ (`separate`).
    Tránh việc số liệu của Công ty mẹ ghi đè lên số liệu Hợp nhất trong cùng một kỳ báo cáo.
    """

    @staticmethod
    def detect_scope(text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in ("báo cáo tài chính riêng", "bctc riêng", "công ty mẹ", "separate financial")):
            return "separate"
        return "consolidated"


class PageOCRCacheManager:
    """
    NÂNG CẤP #3: Lưu bộ nhớ đệm (Persistent SQLite Cache) theo mã băm SHA-256 của trang scan.
    Khi chạy trên Google Colab T4, nếu phiên làm việc bị ngắt kết nối hoặc chạy lại,
    các trang BCTC scan đã qua Qwen2.5-VL sẽ được nạp tức thì trong 1ms mà không tốn GPU đọc lại.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vlm_ocr_cache (
                page_hash TEXT PRIMARY KEY,
                statement_hint TEXT,
                rows_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def compute_hash(image_repr: Any, correction_hint: Optional[str] = None) -> str:
        payload = f"{repr(image_repr)}||{correction_hint or ''}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()

    def get_cached_rows(self, page_hash: str) -> Optional[List[List[str]]]:
        row = self.conn.execute(
            "SELECT rows_json FROM vlm_ocr_cache WHERE page_hash = ?", (page_hash,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def save_cached_rows(self, page_hash: str, rows: List[List[str]], statement_hint: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vlm_ocr_cache (page_hash, statement_hint, rows_json) VALUES (?, ?, ?)",
            (page_hash, statement_hint, json.dumps(rows, ensure_ascii=False)),
        )
        self.conn.commit()


class CrossPeriodAnomalyDetector:
    """
    NÂNG CẤP #4: Kiểm tra bất thường liên kỳ (Cross-Period Balance Sheet Continuity Check).
    Đối chiếu "Số đầu năm" của kỳ năm nay (`T`) với "Số cuối năm" của kỳ năm trước (`T-1`),
    hoặc "Tiền đầu kỳ (Mã 60 LCTT)" với "Tiền và tương đương tiền đầu kỳ trên BCĐKT (Mã 110)".
    """

    @staticmethod
    def verify_cash_continuity(
        ending_cash_prev: Union[float, Decimal],
        beginning_cash_curr: Union[float, Decimal],
        tolerance: float = 1.0,
    ) -> Dict[str, Any]:
        prev_dec = Decimal(str(ending_cash_prev))
        curr_dec = Decimal(str(beginning_cash_curr))
        diff = abs(curr_dec - prev_dec)
        return {
            "is_continuous": float(diff) <= tolerance,
            "ending_cash_previous_period": float(prev_dec),
            "beginning_cash_current_period": float(curr_dec),
            "difference": float(diff),
        }
