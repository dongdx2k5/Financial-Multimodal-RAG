"""Conservative spatial extraction of Vietnamese coded financial statements.

No issuer-specific values. Every accepted value retains its row and PDF coordinates.
"""

import re
import unicodedata
from datetime import date

from .ingestion import _stable_id
from .models import Chunk, Document, PageContent


def fold(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
        if unicodedata.category(c) != "Mn"
    )


METRICS = {
    "income": {
        "01": "Doanh thu bán hàng và cung cấp dịch vụ",
        "02": "Các khoản giảm trừ doanh thu",
        "10": "Doanh thu thuần",
        "11": "Giá vốn hàng bán",
        "20": "Lợi nhuận gộp",
        "22": "Doanh thu hoạt động tài chính",
        "23": "Chi phí tài chính",
        "24": "Chi phí lãi vay",
        "25": "Chi phí bán hàng",
        "26": "Chi phí quản lý doanh nghiệp",
        "30": "Lợi nhuận thuần từ hoạt động kinh doanh",
        "31": "Thu nhập khác",
        "32": "Chi phí khác",
        "40": "Lợi nhuận khác",
        "50": "Lợi nhuận trước thuế",
        "51": "Thuế TNDN hiện hành",
        "52": "Thuế TNDN hoãn lại",
        "60": "Lợi nhuận sau thuế",
    },
    "balance": {
        "100": "Tài sản ngắn hạn",
        "110": "Tiền và các khoản tương đương tiền",
        "120": "Đầu tư tài chính ngắn hạn",
        "130": "Phải thu ngắn hạn",
        "140": "Hàng tồn kho",
        "160": "Tài sản ngắn hạn khác",
        "200": "Tài sản dài hạn",
        "210": "Phải thu dài hạn",
        "220": "Tài sản cố định",
        "240": "Bất động sản đầu tư",
        "250": "Tài sản dở dang dài hạn",
        "260": "Đầu tư tài chính dài hạn",
        "270": "Tài sản dài hạn khác",
        "280": "Tổng tài sản",
        "300": "Nợ phải trả",
        "310": "Nợ ngắn hạn",
        "330": "Nợ dài hạn",
        "400": "Vốn chủ sở hữu",
        "440": "Tổng nguồn vốn",
    },
    "cashflow": {
        "20": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
        "30": "Lưu chuyển tiền thuần từ hoạt động đầu tư",
        "40": "Lưu chuyển tiền thuần từ hoạt động tài chính",
        "50": "Lưu chuyển tiền thuần trong kỳ",
        "60": "Tiền đầu kỳ",
        "61": "Ảnh hưởng thay đổi tỷ giá đối với tiền",
        "70": "Tiền cuối kỳ",
    },
}


def statement_kind(text):
    header = fold(text[:180]).replace(" ", "")
    if "baocaoketqua" in header and "kinhdoanh" in header:
        return "income"
    if "baocaotinhhinh" in header or "bangcandoiketoan" in header:
        return "balance"
    if "baocaoluu" in header:
        return "cashflow"
    return None


def spatial_lines(words):
    groups = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        center = (word[1] + word[3]) / 2
        if groups and abs(center - groups[-1][0]) <= 4:
            groups[-1][1].append(word)
        else:
            groups.append((center, [word]))
    return [(y, sorted(line, key=lambda w: w[0])) for y, line in groups]


def money(raw: str):
    value = raw.replace(" ", "").replace("\u00a0", "")
    if value in {"-", "—", "–"}:
        return None  # A dash is not automatically equated with zero.
    # VND statement columns are integers with groups of exactly 3 digits.
    digits = r"(?:\d{1,3}(?:[.,]\d{3})+|\d+)"
    if not re.fullmatch(rf"(?:-?{digits}|\({digits}\))", value):
        return None
    number = int(value.strip("()").replace(".", "").replace(",", ""))
    return -number if value.startswith("(") else number


def extract_statement(page: PageContent, document: Document):
    kind = statement_kind(page.text)
    if not kind or not page.words or not page.width:
        return [], []
    warnings = []
    headers = []
    for w in page.words:
        match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(20\d{2})", w[4])
        if match and w[0] > page.width * 0.55 and w[1] < page.height * 0.3:
            day, month, year = map(int, match.groups())
            try:
                dt = date(year, month, day)
            except ValueError:
                continue
            headers.append((w, dt))
    headers.sort(key=lambda item: item[0][0])
    if len(headers) != 2:
        return [], [f"Trang {page.page}: chưa xác định chắc chắn hai cột kỳ báo cáo."]
    centers = [(w[0] + w[2]) / 2 for w, _ in headers]
    boundary = sum(centers) / 2
    starts = centers[0] - (centers[1] - centers[0]) * 0.58
    ends = centers[1] + (centers[1] - centers[0]) * 0.6
    header_y = max(w[3] for w, _ in headers)
    header_text = fold(page.text[:650]).replace(" ", "")
    if kind != "balance" and "bathang" not in header_text:
        return [], [
            f"Trang {page.page}: chưa hỗ trợ tự gán kỳ cho bảng không rõ giai đoạn ba tháng."
        ]
    units = ["VND" if re.search(r"\bVND\b", page.text[:1000]) else None] * 2
    if not all(units):
        return [], [f"Trang {page.page}: thiếu đơn vị tiền tệ rõ ràng."]
    lines = spatial_lines(page.words)
    chunks = []
    for row_index, (y, words) in enumerate(lines):
        if y <= header_y + 10 or y > page.height * 0.88:
            continue
        code_words = [
            w
            for w in words
            if page.width * 0.43 < (w[0] + w[2]) / 2 < page.width * 0.50
            and w[4].strip(".") in METRICS[kind]
        ]
        if not code_words:
            code_words = [
                w
                for w in page.secondary_words
                if abs((w[1] + w[3]) / 2 - y) <= 4
                and page.width * 0.43 < (w[0] + w[2]) / 2 < page.width * 0.50
                and w[4].strip(".") in METRICS[kind]
            ]
        code_note = None
        if len(code_words) == 1:
            code = code_words[0][4].strip(".")
        elif not code_words:
            recovered = [r for r in page.recovered_codes if abs(r["y"] - y) <= 4]
            if len(recovered) != 1:
                continue
            code = recovered[0]["code"]
            code_note = f"Mã dòng {code} được đọc lại từ vùng ảnh; chưa xác minh thủ công."
            warnings.append(f"Trang {page.page}: {code_note}")
        else:
            continue
        values = []
        bbox = [
            max(0, min(w[0] for w in words) - 2),
            min(w[1] for w in words) - 3,
            min(page.width, max(w[2] for w in words) + 2),
            max(w[3] for w in words) + 3,
        ]
        for col, (_, dt) in enumerate(headers):
            tokens = [
                w[4]
                for w in words
                if starts <= (w[0] + w[2]) / 2 <= ends
                and (((w[0] + w[2]) / 2 < boundary) == (col == 0))
            ]
            raw = "".join(tokens)
            value = money(raw)
            recognition_note = code_note
            if value is None and page.secondary_words:
                alternative = "".join(
                    w[4]
                    for w in sorted(page.secondary_words, key=lambda w: w[0])
                    if abs((w[1] + w[3]) / 2 - y) <= 4
                    and starts <= (w[0] + w[2]) / 2 <= ends
                    and (((w[0] + w[2]) / 2 < boundary) == (col == 0))
                )
                if money(alternative) is not None:
                    recognition_note = (code_note or "") + (
                        f" OCR bổ sung: {raw} → {alternative}; chưa xác minh thủ công."
                    )
                    raw, value = alternative, money(alternative)
                    warnings.append(f"Trang {page.page}, mã {code}, cột {dt}: {recognition_note}")
            if value is None:
                warnings.append(
                    f"Trang {page.page}, mã {code}, cột {dt}: số OCR không chắc chắn ({raw}); bỏ khỏi phép tính."
                )
                continue
            period = (
                dt.isoformat() if kind == "balance" else f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            )
            values.append(
                {
                    "metric": METRICS[kind][code],
                    "metric_code": code,
                    "value": value,
                    "raw_value": raw,
                    "period": dt.year,
                    "period_label": period,
                    "period_kind": "instant" if kind == "balance" else "quarter",
                    "unit": "VND",
                    "statement": kind,
                    "recognition_note": recognition_note,
                }
            )
        if not values:
            continue
        # Preserve preceding wrapped label for source highlighting.
        if row_index and y - lines[row_index - 1][0] < 20:
            previous = lines[row_index - 1][1]
            if all(w[2] < starts for w in previous):
                bbox[1] = min(bbox[1], min(w[1] for w in previous) - 2)
        body = " | ".join(f"{v['period_label']}: {v['raw_value']} {v['unit']}" for v in values)
        chunks.append(
            Chunk(
                id=_stable_id(document.id, page.page, kind, code, row_index),
                document_id=document.id,
                source=document.source,
                page=page.page,
                ticker=document.ticker,
                year=document.year,
                scope=document.scope,
                text=f"{METRICS[kind][code]} (mã {code}; {document.scope}): {body}",
                bbox=bbox,
                extraction_method=page.extraction_method,
                financial_rows=values,
            )
        )
    return chunks, warnings
