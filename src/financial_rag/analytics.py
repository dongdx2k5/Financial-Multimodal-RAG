"""Deterministic summaries. Missing, ambiguous, or rejected facts are never guessed."""

from collections import Counter

from .calculator import calculate
from .models import FinancialFact

CARDS = [
    ("income", "10"),
    ("income", "60"),
    ("balance", "280"),
    ("balance", "400"),
    ("cashflow", "20"),
    ("cashflow", "70"),
]


def summarize(raw_facts, checks):
    facts = [FinancialFact.model_validate(f) for f in raw_facts]
    cards = []
    for statement, code in CARDS:
        candidates = sorted(
            [f for f in facts if f.statement == statement and f.metric_code == code],
            key=lambda f: f.period_label or "",
            reverse=True,
        )
        if not candidates:
            continue
        latest = candidates[0]
        item = {
            "metric": latest.metric,
            "period": latest.period_label,
            "unit": latest.unit,
            "fact_ids": [latest.id],
            "page": latest.page,
            "chunk_id": latest.chunk_id,
            "value": latest.value,
            "quality": latest.quality,
            "comparison": None,
            "warning": None,
        }
        if sum(f.period_label == latest.period_label for f in candidates) > 1:
            item.update(value=None, warning="Nhiều số liệu cùng kỳ; chưa chọn giá trị.")
        elif latest.quality == "needs_review":
            item.update(value=None, warning="Không khớp kiểm tra; cần đối chiếu nguồn.")
        else:
            older = [f for f in candidates if f.period_label != latest.period_label]
            if older:
                previous = older[0]
                if sum(f.period_label == previous.period_label for f in older) == 1:
                    try:
                        result = calculate("growth", latest, previous)
                        item["comparison"] = {
                            "value": result["value"],
                            "unit": "%",
                            "period": previous.period_label,
                            "formula": result["formula"],
                            "fact_ids": [latest.id, previous.id],
                        }
                    except ValueError as error:
                        item["warning"] = str(error)
        cards.append(item)
    ratios = []
    for code, label in [("20", "Biên lợi nhuận gộp"), ("60", "Biên lợi nhuận ròng")]:
        revenues = [f for f in facts if f.statement == "income" and f.metric_code == "10"]
        for period in sorted({f.period_label for f in revenues if f.period_label}, reverse=True):
            denominator = [f for f in revenues if f.period_label == period]
            numerator = [
                f
                for f in facts
                if f.statement == "income" and f.metric_code == code and f.period_label == period
            ]
            if len(denominator) != 1 or len(numerator) != 1:
                continue
            try:
                calculation = calculate("ratio", numerator[0], denominator[0])
                ratios.append(
                    {
                        "metric": label,
                        "period": period,
                        "value": calculation["value"],
                        "unit": "%",
                        "formula": calculation["formula"],
                        "fact_ids": [numerator[0].id, denominator[0].id],
                    }
                )
            except ValueError:
                continue
    return {
        "cards": cards,
        "ratios": ratios,
        "checks": dict(Counter(c["status"] for c in checks)),
        "warning": "Chỉ tóm tắt dữ liệu trích xuất, không phải xác nhận kiểm toán hoặc khuyến nghị đầu tư.",
    }
