"""Regression tests for AI-introduced blind spots and edge cases.

Complies with .agents/skills/ai-regression-testing/SKILL.md guidelines.
"""

import tempfile
from pathlib import Path
import pytest

from financial_rag.db.schema import create_database
from financial_rag.db.repository import FinancialRepository
from financial_rag.db.seed import seed_database
from financial_rag.parser.ingest_pipeline import ingest_single_file


def test_seed_idempotency_regression():
    """BUG-R1: Calling seed_database multiple times must NOT duplicate reports or questions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        db_path = tmp_p / "test_seed.db"
        manifest_path = tmp_p / "manifest.csv"
        metrics_path = tmp_p / "metrics.csv"
        eval_path = tmp_p / "eval.csv"

        # Create dummy manifest with an extra trailing blank line
        manifest_path.write_text(
            "ticker,company_name_vi,company_name_en,industry,country,doc_type,period_type,fiscal_year,period_end,scope,language,currency,unit_scale,audit_status,raw_file_path,is_scanned\n"
            "TEST,Công ty Test,Test Corp,IT,VN,BCTC,FY,2025,2025-12-31,consolidated,vi,VND,1,audited,data/raw/test.txt,false\n\n",
            encoding="utf-8"
        )
        metrics_path.write_text(
            "code,name_vi,name_en,statement,is_common,aliases\n"
            "net_revenue,Doanh thu thuần,Net revenue,IS,1,Doanh thu\n",
            encoding="utf-8"
        )
        eval_path.write_text(
            "question,language,difficulty,gold_answer,gold_report_id,gold_page\n"
            "Doanh thu bao nhiêu?,vi,lookup,100,1,1\n",
            encoding="utf-8"
        )

        # Run seed first time
        seed_database(db_path=db_path, manifest_path=manifest_path, metrics_path=metrics_path, eval_path=eval_path)
        repo = FinancialRepository(db_path)
        assert len(repo.find_reports(ticker="TEST")) == 1
        assert len(repo.get_all_eval_questions()) == 1

        # Run seed second time (MUST NOT duplicate)
        seed_database(db_path=db_path, manifest_path=manifest_path, metrics_path=metrics_path, eval_path=eval_path)
        assert len(repo.find_reports(ticker="TEST")) == 1
        assert len(repo.get_all_eval_questions()) == 1


def test_company_name_preservation_regression():
    """BUG-R2: Ingesting file without explicit company name must NOT overwrite existing company name with ticker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        db_path = tmp_p / "test.db"
        create_database(db_path)
        repo = FinancialRepository(db_path)

        # Add company with full Vietnamese and English names
        repo.add_company(
            ticker="NVDA",
            name_vi="Tập đoàn NVIDIA",
            name_en="NVIDIA Corporation",
            industry="Bán dẫn",
            country="US"
        )

        # Create dummy report file
        dummy_file = tmp_p / "nvda_report.txt"
        dummy_file.write_text(
            "===== PAGE 1 =====\nBáo cáo tài chính năm 2025\nDoanh thu thuần: 60.922 triệu USD\n",
            encoding="utf-8"
        )

        # Ingest without providing company_name_vi (defaults should NOT wipe existing names)
        res = ingest_single_file(
            file_path=dummy_file,
            ticker="NVDA",
            fiscal_year=2025,
            period_type="FY",
            repo=repo,
            update_manifest=False,
        )
        assert res["status"] == "success"

        comp = repo.get_company_by_ticker("NVDA")
        assert comp is not None
        # Assert full names are PRESERVED and not wiped to 'NVDA'
        assert comp["name_vi"] == "Tập đoàn NVIDIA"
        assert comp["name_en"] == "NVIDIA Corporation"


def test_ingest_failure_atomicity():
    """BUG-R3: If parsing fails, no dangling report should be committed to the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        db_path = tmp_p / "test.db"
        create_database(db_path)
        repo = FinancialRepository(db_path)

        # File with unsupported extension that will fail parser
        corrupt_file = tmp_p / "invalid_file.xyz"
        corrupt_file.write_text("not a pdf or text file", encoding="utf-8")

        with pytest.raises(Exception):
            ingest_single_file(
                file_path=corrupt_file,
                ticker="FAIL",
                fiscal_year=2025,
                repo=repo,
                update_manifest=False,
            )

        # Verify no report was left in DB
        reports = repo.find_reports(ticker="FAIL")
        assert len(reports) == 0


def test_ingest_single_file_idempotency():
    """BUG-R4: Re-ingesting the same report should update facts/chunks cleanly without uniqueness errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        db_path = tmp_p / "test.db"
        create_database(db_path)
        repo = FinancialRepository(db_path)

        test_file = tmp_p / "test_report.txt"
        test_file.write_text(
            "===== PAGE 1 =====\nDoanh thu bán hàng và cung cấp dịch vụ: 500 tỷ\n",
            encoding="utf-8"
        )

        # First ingest
        res1 = ingest_single_file(
            file_path=test_file,
            ticker="DEMO",
            fiscal_year=2025,
            repo=repo,
            update_manifest=False,
        )
        assert res1["status"] == "success"
        facts1 = repo.query_facts(ticker="DEMO", fiscal_year=2025)

        # Second ingest of same report (clean re-ingest)
        res2 = ingest_single_file(
            file_path=test_file,
            ticker="DEMO",
            fiscal_year=2025,
            repo=repo,
            update_manifest=False,
        )
        assert res2["status"] == "success"
        facts2 = repo.query_facts(ticker="DEMO", fiscal_year=2025)

        # Must have same number of facts, not doubled!
        assert len(facts1) == len(facts2)
