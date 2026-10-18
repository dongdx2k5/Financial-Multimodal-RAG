"""Tests for Hybrid retrieval fusing dense embeddings and lexical FTS5."""

import tempfile
from pathlib import Path
from financial_rag.db.schema import create_database
from financial_rag.db.repository import FinancialRepository
from financial_rag.retrieval.hybrid import HybridRetriever


def test_hybrid_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_rag.db"
        create_database(db_path)
        repo = FinancialRepository(db_path)

        c_id = repo.add_company(ticker="AAA", name_vi="Nhựa An Phát Xanh")
        r_id = repo.add_report(company_id=c_id, fiscal_year=2025, period_type="FY")

        # Add metric with alias
        repo.add_metric(
            code="net_revenue",
            name_vi="Doanh thu thuần",
            name_en="Net revenue",
            aliases="Doanh thu thuần|Net revenue|Doanh thu bán hàng",
        )

        repo.add_chunk(
            report_id=r_id,
            section="KQKD",
            chunk_type="table",
            page_start=7,
            page_end=7,
            context_prefix="[AAA | FY/2025]",
            content="Doanh thu bán hàng và cung cấp dịch vụ tăng trưởng mạnh nhờ xuất khẩu bao bì phân hủy sinh học.",
        )
        repo.add_chunk(
            report_id=r_id,
            section="Ý kiến kiểm toán",
            chunk_type="text",
            page_start=4,
            page_end=4,
            context_prefix="[AAA | FY/2025]",
            content="Ý kiến kiểm toán viên: Báo cáo tài chính phản ánh trung thực và hợp lý trên các khía cạnh trọng yếu.",
        )

        retriever = HybridRetriever(
            repo=repo,
            dense_weight=0.5,
            fts_weight=0.5,
            relevance_threshold=0.2,
        )
        # Build index
        retriever.dense_retriever.index_path = Path(tmpdir) / "faiss.index"
        retriever.dense_retriever.chunk_ids_path = Path(tmpdir) / "chunk_ids.json"
        retriever.dense_retriever.build_index()

        results = retriever.retrieve("ý kiến kiểm toán", ticker="AAA", top_k=2)
        assert len(results) >= 1
        assert "kiểm toán" in results[0].content.lower()
