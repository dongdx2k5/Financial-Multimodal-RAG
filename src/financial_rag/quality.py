from calendar import monthrange
from collections import defaultdict
from decimal import Decimal

RULES = {
    "income": [
        ("10", {"01": 1, "02": -1}),
        ("20", {"10": 1, "11": -1}),
        ("30", {"20": 1, "22": 1, "23": -1, "25": -1, "26": -1}),
        ("40", {"31": 1, "32": -1}),
        ("50", {"30": 1, "40": 1}),
        ("60", {"50": 1, "51": -1, "52": -1}),
    ],
    "balance": [
        ("280", {"100": 1, "200": 1}),
        ("300", {"310": 1, "330": 1}),
        ("440", {"300": 1, "400": 1}),
        ("280", {"440": 1}),
    ],
    "cashflow": [("50", {"20": 1, "30": 1, "40": 1}), ("70", {"50": 1, "60": 1, "61": 1})],
}


def check_facts(facts):
    groups = defaultdict(lambda: defaultdict(list))
    for fact in facts:
        if fact.metric_code:
            groups[
                (
                    fact.document_id or fact.source,
                    fact.scope,
                    fact.period_kind,
                    fact.statement,
                    fact.period_label,
                )
            ][fact.metric_code].append(fact)
    checks = []
    for (_, _, _, statement, period), candidates in groups.items():
        mapping = {code: items[0] for code, items in candidates.items() if len(items) == 1}
        for target, terms in RULES.get(statement, []):
            codes = [target, *terms]
            missing = [code for code in codes if code not in mapping]
            item = {
                "statement": statement,
                "period": period,
                "target_code": target,
                "formula": f"{target} = "
                + " ".join(f"{coef:+d}*{code}" for code, coef in terms.items()),
            }
            if missing:
                ambiguous = [code for code in missing if code in candidates]
                for code in ambiguous:
                    for fact in candidates[code]:
                        fact.quality = "needs_review"
                checks.append(
                    {
                        **item,
                        "status": "incomplete",
                        "missing_codes": missing,
                        "ambiguous_codes": ambiguous,
                    }
                )
                continue
            expected = sum(Decimal(str(mapping[code].value)) * coef for code, coef in terms.items())
            actual = Decimal(str(mapping[target].value))
            ok = actual == expected
            checks.append(
                {
                    **item,
                    "status": "pass" if ok else "fail",
                    "difference": str(actual - expected),
                    "fact_ids": [mapping[code].id for code in codes],
                }
            )
            for code in codes:
                fact = mapping[code]
                if not ok:
                    fact.quality = "needs_review"
                elif fact.quality != "needs_review":
                    fact.quality = "consistent"  # Arithmetic consistency is not human verification.
    # Match only the exact cash-flow opening/closing dates, entity, and report scope.
    for fact in facts:
        if (
            fact.statement != "cashflow"
            or fact.metric_code not in {"60", "70"}
            or fact.period_kind != "quarter"
            or not fact.period_label
        ):
            continue
        year, quarter = fact.period_label.split("-Q")
        month = (int(quarter) - 1) * 3 + 1 if fact.metric_code == "60" else int(quarter) * 3
        day = 1 if fact.metric_code == "60" else monthrange(int(year), month)[1]
        date_label = f"{year}-{month:02d}-{day:02d}"
        matches = [
            other
            for other in facts
            if other.statement == "balance"
            and other.metric_code == "110"
            and other.period_label == date_label
            and (other.document_id or other.source) == (fact.document_id or fact.source)
            and other.scope == fact.scope
        ]
        item = {
            "statement": "cross_statement",
            "period": fact.period_label,
            "target_code": fact.metric_code,
            "formula": f"cashflow.{fact.metric_code} = balance.110 ({date_label})",
        }
        if len(matches) != 1:
            checks.append(
                {
                    **item,
                    "status": "incomplete",
                    "missing_codes": ["balance.110"],
                    "reason": "Thiếu hoặc trùng số dư tiền đúng ngày đối chiếu.",
                }
            )
            continue
        other = matches[0]
        delta = Decimal(str(fact.value)) - Decimal(str(other.value))
        ok = delta == 0 and fact.unit == other.unit
        checks.append(
            {
                **item,
                "status": "pass" if ok else "fail",
                "difference": str(delta),
                "fact_ids": [fact.id, other.id],
            }
        )
        for involved in [fact, other]:
            if not ok:
                involved.quality = "needs_review"
            elif involved.quality != "needs_review":
                involved.quality = "consistent"
    return checks
