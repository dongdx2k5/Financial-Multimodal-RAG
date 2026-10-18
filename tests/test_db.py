"""Tests for SQLite database schema, repository operations and FTS5 search."""

import tempfile
from pathlib import Path
from financial_rag.db.schema import create_database
from financial_rag.db.repository import FinancialRepository


def test_db_crud_and_fts():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_financial.db"
        create_database(db_path)
        repo = FinancialRepository(db_path)

        # 1. Company
        c_id = repo.add_company(ticker="AAA", name_vi="Nhựa An Phát Xanh", industry="Nhựa")
        assert c_id > 0
        comp = repo.get_company_by_ticker("AAA")
        assert comp is not None
        assert comp["ticker"] == "AAA"

        # 2. Metric
        m_id = repo.add_metric(
            code="net_revenue",
            name_vi="Doanh thu thuần",
            name_en="Net revenue",
            statement="IS",
            aliases="Doanh thu thuần|Doanh thu bán hàng|Net revenue",
        )
        assert m_id > 0
        matched = repo.match_metric_alias("Doanh thu bán hàng")
        assert matched is not None
        assert matched["code"] == "net_revenue"

        # 3. Report
        r_id = repo.add_report(
            company_id=c_id,
            fiscal_year=2025,
            period_type="FY",
            currency="VND",
            unit_scale=1,
        )
        assert r_id > 0

        # 4. Fact
        f_id = repo.add_fact(
            report_id=r_id,
            line_item="Doanh thu thuần",
            column_role="current",
            value=4225864218423.0,
            value_raw="4.225.864.218.423",
            metric_id=m_id,
            page=7,
        )
        assert f_id > 0
        queried_facts = repo.query_facts(ticker="AAA", fiscal_year=2025, metric_code="net_revenue")
        assert len(queried_facts) == 1
        assert queried_facts[0]["value"] == 4225864218423.0

        # 5. Chunk & FTS5
        ch_id = repo.add_chunk(
            report_id=r_id,
            section="Báo cáo kết quả kinh doanh",
            chunk_type="table",
            page_start=7,
            page_end=7,
            context_prefix="[AAA | FY/2025 | Riêng lẻ]",
            content="Tổng doanh thu thuần về bán hàng và cung cấp dịch vụ năm 2025 đạt 4.225 tỷ đồng.",
        )
        assert ch_id > 0

        # Search FTS5
        search_res = repo.search_chunks_fts("doanh thu thuần", ticker="AAA")
        assert len(search_res) >= 1
        assert search_res[0]["chunk_id"] == ch_id
