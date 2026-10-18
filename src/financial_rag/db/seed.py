"""Seed data initialization script for Financial RAG."""

import csv
from pathlib import Path
from typing import Optional, Union
from financial_rag.config import settings
from financial_rag.db.schema import create_database
from financial_rag.db.repository import FinancialRepository


def seed_database(
    db_path: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    metrics_path: Optional[Union[str, Path]] = None,
    eval_path: Optional[Union[str, Path]] = None,
) -> None:
    """Populate database with metrics dictionary, manifest reports, and eval questions."""
    db_p = Path(db_path) if db_path else settings.resolve_path(settings.database_path)
    man_p = Path(manifest_path) if manifest_path else settings.resolve_path(settings.manifest_path)
    met_p = Path(metrics_path) if metrics_path else settings.resolve_path(settings.metrics_seed_path)
    ev_p = Path(eval_path) if eval_path else settings.resolve_path(settings.eval_questions_path)

    # Initialize schema
    create_database(db_p)
    repo = FinancialRepository(db_p)

    # 1. Seed Metrics
    if met_p.exists():
        with open(met_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get("code") or not row.get("code").strip():
                    continue
                repo.add_metric(
                    code=row["code"].strip(),
                    name_vi=row.get("name_vi", "").strip(),
                    name_en=row.get("name_en", "").strip(),
                    statement=row.get("statement", "IS"),
                    is_common=bool(int(row.get("is_common", 1))),
                    aliases=row.get("aliases"),
                )

    # 2. Seed Manifest (Companies & Reports)
    if man_p.exists():
        with open(man_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get("ticker") or not row.get("ticker").strip():
                    continue
                ticker = row["ticker"].strip().upper()
                fy = int(row["fiscal_year"])
                pt = (row.get("period_type") or "FY").strip().upper()
                sc = (row.get("scope") or "consolidated").strip().lower()

                company_id = repo.add_company(
                    ticker=ticker,
                    name_vi=row.get("company_name_vi"),
                    name_en=row.get("company_name_en"),
                    industry=row.get("industry"),
                    country=row.get("country"),
                )

                # Avoid duplicate report inserts when seed is run multiple times
                existing = [
                    r for r in repo.find_reports(
                        ticker=ticker,
                        fiscal_year=fy,
                        period_type=pt
                    )
                    if (r.get("scope") or "").strip().lower() == sc
                ]
                if not existing:
                    repo.add_report(
                        company_id=company_id,
                        doc_type=row.get("doc_type", "BCTC"),
                        period_type=pt,
                        fiscal_year=fy,
                        period_end=row.get("period_end"),
                        scope=sc,
                        language=row.get("language", "vi"),
                        currency=(row.get("currency") or "VND").strip().upper(),
                        unit_scale=int(row.get("unit_scale", 1)),
                        audit_status=row.get("audit_status", "audited"),
                        file_path=row.get("raw_file_path"),
                        is_scanned=row.get("is_scanned", "").lower() in ("true", "1", "yes"),
                    )

    # 3. Seed Evaluation Questions
    if ev_p.exists():
        existing_qs = repo.get_all_eval_questions()
        if not existing_qs:
            with open(ev_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row or not row.get("question") or not row.get("question").strip():
                        continue
                    gold_rep = int(row["gold_report_id"]) if row.get("gold_report_id") and row["gold_report_id"].isdigit() else None
                    gold_p = int(row["gold_page"]) if row.get("gold_page") and row["gold_page"].isdigit() else None
                    repo.add_eval_question(
                        question=row["question"].strip(),
                        language=(row.get("language") or "vi").strip().lower(),
                        difficulty=(row.get("difficulty") or "lookup").strip().lower(),
                        gold_answer=row.get("gold_answer"),
                        gold_report_id=gold_rep,
                        gold_page=gold_p,
                    )


if __name__ == "__main__":
    seed_database()
    print("Database seeded successfully.")
