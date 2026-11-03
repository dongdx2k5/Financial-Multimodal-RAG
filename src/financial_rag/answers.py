"""Evidence-bound answering and explicit arithmetic for supported financial intents."""

import re
from decimal import Decimal

from .calculator import calculate
from .extraction import parse_vietnamese_number
from .financial_tables import fold, money
from .models import AskResponse, Evidence, FinancialFact

ALIASES = {
    "lnst": "loi nhuan sau thue",
    "lai rong": "loi nhuan sau thue",
    "loi nhuan rong": "loi nhuan sau thue",
    "doanh thu thuan": "doanh thu thuan",
    "bien lai rong": "bien loi nhuan rong",
    "ln gop": "loi nhuan gop",
    "tong cong tai san": "tong tai san",
    "von csh": "von chu so huu",
    "dong tien tu hoat dong kinh doanh": "dong tien kinh doanh",
    "luu chuyen tien thuan tu hoat dong kinh doanh": "dong tien kinh doanh",
    "dong tien tu hoat dong dau tu": "dong tien dau tu",
    "luu chuyen tien thuan tu hoat dong dau tu": "dong tien dau tu",
    "dong tien tu hoat dong tai chinh": "dong tien tai chinh",
    "luu chuyen tien thuan tu hoat dong tai chinh": "dong tien tai chinh",
}
INTENTS = [
    ("loi nhuan sau thue", "income", "60"),
    ("loi nhuan truoc thue", "income", "50"),
    ("loi nhuan gop", "income", "20"),
    ("doanh thu thuan", "income", "10"),
    ("doanh thu hoat dong tai chinh", "income", "22"),
    ("doanh thu ban hang", "income", "01"),
    ("doanh thu", "income", "10"),
    ("gia von", "income", "11"),
    ("tong tai san", "balance", "280"),
    ("von chu so huu", "balance", "400"),
    ("no phai tra", "balance", "300"),
    ("hang ton kho", "balance", "140"),
    ("dong tien kinh doanh", "cashflow", "20"),
    ("dong tien dau tu", "cashflow", "30"),
    ("dong tien tai chinh", "cashflow", "40"),
    ("tien dau ky", "cashflow", "60"),
    ("tien cuoi ky", "cashflow", "70"),
]


def expand_query(question):
    result = fold(question)
    for alias, full in ALIASES.items():
        result = re.sub(r"\b" + re.escape(alias) + r"\b", full, result)
    return result


def number(value):
    value = Decimal(str(value))
    return decimal_vi(value, max(0, -value.normalize().as_tuple().exponent))


def decimal_vi(value, places=2):
    return (
        f"{Decimal(str(value)):,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    )


def clarification(message, reason):
    return AskResponse(
        answer=message,
        evidence=[],
        mode="evidence",
        answer_status="needs_clarification",
        reason_code=reason,
    )


def supported_wording(q, matches, ticker=None):
    """Conservative vocabulary gate; never answer just the recognized half of a question."""
    remainder = q
    for phrase, _, _ in sorted(matches, key=lambda item: len(item[0]), reverse=True):
        remainder = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", remainder)
    if ticker:
        remainder = re.sub(r"\b" + re.escape(fold(ticker)) + r"\b", " ", remainder)
    remainder = re.sub(r"\b\d{1,2}/\d{1,2}/20\d{2}\b", " ", remainder)
    remainder = re.sub(r"\bquy\s*(iv|iii|ii|i|[1-4])\b", " ", remainder)
    remainder = re.sub(r"\b20\d{2}\b", " ", remainder)
    allowed_words = (
        "la bao nhieu vao tai ngay nam quy ky dau cuoi cung so voi tang giam "
        "truong chenh lech sanh phan tram rieng hop nhat bao cao cua cong ty "
        "hay cho biet tinh gia tri bien ca toan trong va"
    )
    return set(re.findall(r"\w+", remainder)).issubset(allowed_words.split())


def matches_numeric_source(fact, chunk):
    """Check that the selected value still agrees with the saved source literal."""
    raw = re.sub(r"\s+", "", fact.raw_value)
    if not raw:
        return False
    try:
        parsed = money(raw)
        if parsed is None:
            parsed = parse_vietnamese_number(raw)
        value = Decimal(str(parsed))
    except (ValueError, ArithmeticError):
        return False
    text = re.sub(r"\s+", "", chunk.text)
    return (
        value.is_finite()
        and value == Decimal(str(fact.value))
        and re.search(r"(?<![\d.,])" + re.escape(raw) + r"(?![\d.,])", text) is not None
    )


def factual_answer(service, question, document_id):
    if not document_id:
        return None  # Never silently combine different reports.
    q = expand_query(question)
    if not any(c.document_id == document_id and c.financial_rows for c in service.retriever.chunks):
        return None
    margin = "bien" in q and ("loi nhuan" in q or "lai" in q)
    matches = [(phrase, statement, code) for phrase, statement, code in INTENTS if phrase in q]
    matches = [
        item
        for item in matches
        if not any(item[0] != other[0] and item[0] in other[0] for other in matches)
    ]
    intent = (matches[0][1], matches[0][2]) if matches else None
    requested = {(statement, code) for _, statement, code in matches}
    if len(requested) > 1:
        return None
    if margin:
        # Only gross/net margins are implemented; do not silently map every margin to net.
        if not ({"20", "60"} & {code for _, statement, code in matches if statement == "income"}):
            return None
        intent = ("income", "20" if "gop" in q else "60")
    if not intent:
        return None
    # Qualitative questions are retrieval-only, never generated by a language model.
    if any(
        term in q
        for term in [
            "tai sao",
            "vi sao",
            "nguyen nhan",
            "rui ro",
            "trien vong",
            "phan tich",
            "chinh sach",
            "ghi nhan",
            "khi nao",
            "dieu kien",
            "tom tat",
            "quy dinh",
        ]
    ):
        return None
    report = service.graph.graph.nodes.get(document_id, {})
    if not supported_wording(q, matches, report.get("ticker")):
        return None
    if "hop nhat" in q and "rieng" in q:
        return clarification(
            "Hãy chọn một phạm vi: báo cáo riêng hoặc hợp nhất.", "multiple_scopes"
        )
    if margin and any(term in q for term in ["tang", "giam", "chenh lech", "so voi", "so sanh"]):
        return clarification(
            "Hiện chỉ tính biên lợi nhuận của một kỳ; chưa tính thay đổi biên giữa hai kỳ.",
            "margin_comparison_unsupported",
        )
    all_facts = [
        FinancialFact(id=node, **{k: v for k, v in attrs.items() if k != "kind"})
        for node, attrs in service.graph.graph.nodes(data=True)
        if attrs.get("kind") == "Fact" and attrs.get("document_id") == document_id
    ]
    facts = [f for f in all_facts if (f.statement, f.metric_code) == intent]
    if not facts:
        return AskResponse(
            answer="Chưa có số liệu trích xuất đủ chắc chắn cho chỉ tiêu này. Cần kiểm tra trang gốc.",
            evidence=[],
            mode="evidence",
            warnings=["Không dùng số OCR thiếu hoặc không hợp lệ để tính."],
        )
    if any(not f.period_label or f.period is None for f in facts):
        return AskResponse(
            answer="Dữ kiện chỉ tiêu thiếu kỳ; chưa thể chọn kỳ mới nhất hoặc kỳ được hỏi.",
            evidence=[],
            mode="evidence",
            reason_code="incomplete_metadata",
        )
    if ("hop nhat" in q and facts[0].scope != "consolidated") or (
        "rieng" in q and facts[0].scope != "separate"
    ):
        return AskResponse(
            answer="Phạm vi câu hỏi không khớp báo cáo đã chọn. Đây là báo cáo "
            + facts[0].scope
            + ".",
            evidence=[],
            mode="evidence",
        )
    years = re.findall(r"\b20\d{2}\b", q)
    quarters = re.findall(r"quy\s*(iv|iii|ii|i|[1-4])\b", q)
    quarter_values = {{"i": "1", "ii": "2", "iii": "3", "iv": "4"}.get(v, v) for v in quarters}
    if len(quarter_values) > 1:
        return clarification(
            "Chưa hỗ trợ so sánh khác quý. Hãy hỏi một quý hoặc so sánh cùng quý giữa hai năm.",
            "multiple_quarters",
        )
    quarter = re.search(r"quy\s*(iv|iii|ii|i|[1-4])\b", q)
    if quarter and any(term in q for term in ["ca nam", "toan nam"]):
        return clarification("Hãy chọn kỳ quý hoặc cả năm, không dùng đồng thời.", "mixed_periods")
    if (
        any(term in q for term in ["dau nam", "dau ky"])
        and "cuoi ky" in q
        and not any(term in q for term in ["so voi", "chenh lech", "so sanh", "tang", "giam"])
    ):
        return clarification(
            "Hãy chọn đầu kỳ hoặc cuối kỳ, hoặc nêu rõ phép so sánh.", "mixed_periods"
        )
    if quarter:
        quarter_value = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}.get(quarter[1], quarter[1])
        if any(f.period_kind != "quarter" for f in facts):
            return clarification(
                "Chỉ tiêu là số dư tại ngày. Hãy ghi ngày cụ thể hoặc đầu năm/cuối kỳ.",
                "instant_requires_date",
            )
        facts = [f for f in facts if (f.period_label or "").endswith("Q" + quarter_value)]
    growth = any(
        term in q for term in ["tang", "giam", "so voi", "cung ky", "chenh lech", "so sanh"]
    )
    if not growth and len(set(years)) > 1:
        return clarification(
            "Hãy hỏi một kỳ hoặc nêu rõ yêu cầu so sánh hai kỳ.", "multiple_periods"
        )
    if years and not growth:
        facts = [f for f in facts if str(f.period) in years]
    if not facts:
        return AskResponse(
            answer="Không tìm thấy chỉ tiêu đúng kỳ được yêu cầu trong báo cáo này.",
            evidence=[],
            mode="evidence",
        )
    if (
        not quarter
        and any(f.period_kind == "quarter" for f in facts)
        and (years or "ca nam" in q or "toan nam" in q)
    ):
        return clarification(
            "Báo cáo chỉ có số phát sinh theo quý, không phải cả năm. "
            "Hãy ghi rõ quý cần hỏi; không dùng số một quý thay cho số cả năm.",
            "quarter_required",
        )
    dates = re.findall(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", q)
    if dates and any(f.period_kind != "instant" for f in facts):
        return clarification(
            "Chỉ tiêu phát sinh cần kỳ báo cáo, không phải một ngày. Hãy ghi rõ quý.",
            "period_not_date",
        )
    if dates:
        wanted = {f"{int(y):04d}-{int(m):02d}-{int(d):02d}" for d, m, y in dates}
        if len(wanted) != (2 if growth else 1):
            return clarification(
                "Hãy nêu đúng một ngày để tra số dư hoặc hai ngày để so sánh.",
                "ambiguous_dates",
            )
        facts = [f for f in facts if f.period_label in wanted]
    if growth and len(set(years)) == 2 and not dates:
        facts = [f for f in facts if str(f.period) in years]
    if not facts:
        return AskResponse(
            answer="Không có số liệu đúng ngày yêu cầu.", evidence=[], mode="evidence"
        )
    # Any ambiguous duplicate at the same period must be resolved, not selected arbitrarily.
    period_counts = {
        p: sum(f.period_label == p for f in facts) for p in {f.period_label for f in facts}
    }
    if any(count > 1 for count in period_counts.values()):
        return AskResponse(
            answer="Có nhiều số liệu cùng chỉ tiêu và kỳ; cần đối chiếu trước khi trả lời.",
            evidence=[],
            mode="evidence",
        )
    facts.sort(key=lambda f: f.period_label or "", reverse=True)
    if (
        intent != ("cashflow", "60")
        and not growth
        and not margin
        and any(term in q for term in ["dau nam", "dau ky"])
    ):
        if any(f.period_kind != "instant" for f in facts):
            return AskResponse(
                answer="Chỉ tiêu này là số phát sinh trong kỳ, không phải số dư đầu kỳ.",
                evidence=[],
                mode="evidence",
            )
        # Beginning of year is specifically January 1, not merely the oldest observation.
        if "dau nam" in q:
            facts = [f for f in facts if (f.period_label or "").endswith("-01-01")]
        if not facts:
            return AskResponse(
                answer="Không có số dư đầu năm trong dữ liệu.", evidence=[], mode="evidence"
            )
        facts.sort(key=lambda f: f.period_label or "")
    chosen = facts[:2] if growth else facts[:1]
    if growth and len(set(years)) > 2:
        return clarification("Hiện chỉ so sánh hai kỳ trong một lần hỏi.", "too_many_periods")
    if growth and years and not set(years).issubset({str(f.period) for f in chosen}):
        return AskResponse(
            answer="Không đủ các năm cần so sánh trong báo cáo.", evidence=[], mode="evidence"
        )
    if growth and len(set(years)) == 1 and str(chosen[0].period) != years[0]:
        return clarification(
            "Không có đủ dữ liệu để lấy năm yêu cầu làm kỳ so sánh mới.", "wrong_target_year"
        )
    if growth and len(set(years)) == 2 and "so voi" in q and years[0] != str(chosen[0].period):
        return clarification(
            "Hiện tính kỳ mới so với kỳ cũ. Hãy đặt kỳ mới trước trong câu hỏi.",
            "reverse_comparison",
        )
    if margin:
        revenue = [
            f
            for f in all_facts
            if f.statement == "income"
            and f.metric_code == "10"
            and f.period_label == chosen[0].period_label
        ]
        if len(revenue) != 1:
            return AskResponse(
                answer="Thiếu doanh thu cùng kỳ để tính biên lợi nhuận.",
                evidence=[],
                mode="evidence",
            )
        chosen = [chosen[0], revenue[0]]
    evidence = []
    for fact in chosen:
        chunk = next((c for c in service.retriever.chunks if c.id == fact.chunk_id), None)
        if (
            chunk is None
            or chunk.document_id != document_id
            or chunk.page != fact.page
            or chunk.source != fact.source
        ):
            return AskResponse(
                answer="Không đối chiếu được dữ kiện với chunk nguồn. Cần nạp lại báo cáo.",
                evidence=[],
                mode="evidence",
                reason_code="invalid_provenance",
            )
        evidence.append(
            Evidence(
                chunk_id=chunk.id,
                source=chunk.source,
                page=chunk.page,
                text=chunk.text,
                score=1,
                graph_facts=[fact],
                document_id=document_id,
                source_url=f"/documents/{document_id}/file#page={chunk.page}",
                bbox=chunk.bbox,
                extraction_method=chunk.extraction_method,
            )
        )
        if not matches_numeric_source(fact, chunk):
            return AskResponse(
                answer="Giá trị không khớp chuỗi số trong bằng chứng đã lưu. "
                "Cần đối chiếu và nạp lại tài liệu trước khi trả lời.",
                evidence=evidence,
                mode="evidence",
                reason_code="source_value_mismatch",
            )
    warnings = ["Kiểm tra khớp công thức không thay thế xác minh trang gốc."]
    if any(e.extraction_method == "ocr" for e in evidence):
        warnings.append("Số liệu được nhận dạng bằng OCR, có thể đọc sai chữ hoặc số.")
    if not years and not dates and not any(term in q for term in ["dau nam", "dau ky"]):
        warnings.append("Chưa chỉ định kỳ cụ thể: dùng kỳ mới nhất phù hợp trong báo cáo.")
    if any(
        not f.unit
        or f.scope not in {"separate", "consolidated"}
        or not f.period_label
        or f.period is None
        or f.period_kind not in {"instant", "quarter", "year"}
        or not Decimal(str(f.value)).is_finite()
        for f in chosen
    ):
        return AskResponse(
            answer="Dữ kiện thiếu đơn vị, kỳ hoặc phạm vi đã xác định; chưa đủ điều kiện trả số.",
            evidence=evidence,
            mode="evidence",
            reason_code="incomplete_metadata",
            warnings=warnings,
        )
    if any(f.quality == "needs_review" for f in chosen):
        return AskResponse(
            answer="Số liệu không khớp kiểm tra kế toán. Cần đối chiếu nguồn trước khi kết luận.",
            evidence=evidence,
            mode="evidence",
            warnings=warnings,
        )
    scope = "riêng" if chosen[0].scope == "separate" else chosen[0].scope
    if growth or margin:
        if len(chosen) != 2:
            return AskResponse(
                answer="Chưa đủ hai kỳ để tính tăng trưởng.", evidence=evidence, mode="evidence"
            )
        operation = (
            "ratio"
            if margin
            else (
                "difference"
                if "chenh lech" in q and "%" not in q and "phan tram" not in q
                else "growth"
            )
        )
        try:
            result = calculate(operation, *chosen)
        except ValueError as exc:
            return AskResponse(
                answer=str(exc), evidence=evidence, mode="evidence", warnings=warnings
            )
        value = Decimal(result["value"])
        label = (
            "Biên lợi nhuận"
            if margin
            else ("Chênh lệch" if operation == "difference" else "Thay đổi")
        )
        metric_label = chosen[0].metric.lower()
        if margin:
            metric_label = metric_label.removeprefix("lợi nhuận ")
        text = (
            f"{label} {metric_label}: {decimal_vi(value)} {result['unit']}. "
            f"Báo cáo {scope} [1] [2].\n"
        )
        text += "\n".join(
            f"[{i}] {f.metric} — {f.period_label}: {number(f.value)} {f.unit} (trang PDF {f.page})."
            for i, f in enumerate(chosen, 1)
        )
        text += (
            "\nCông thức: "
            + result["formula"]
            + " = "
            + str(value.quantize(Decimal("0.01")))
            + " "
            + result["unit"]
        )
        return AskResponse(answer=text, evidence=evidence, mode="calculation", warnings=warnings)
    f = chosen[0]
    converted = (
        f" (≈ {decimal_vi(Decimal(str(f.value)) / Decimal(10**9), 3)} tỷ đồng)"
        if f.unit in {"VND", "đồng"}
        else ""
    )
    return AskResponse(
        answer=f"{f.metric} — {f.period_label}, báo cáo {scope}: {number(f.value)} {f.unit}"
        f"{converted} [1]. Trang PDF {f.page}.",
        evidence=evidence,
        mode="structured",
        warnings=warnings,
    )
