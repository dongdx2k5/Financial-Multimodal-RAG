"""End-to-end tests for the complete Financial RAG pipeline."""

import tempfile
from pathlib import Path
from financial_rag.db.schema import create_database
from financial_rag.db.repository import FinancialRepository
from financial_rag.llm.model import LLMBackend
from financial_rag.reasoning.engine import FinancialRAGEngine
from financial_rag.retrieval.hybrid import HybridRetriever


def test_rag_pipeline_lookup_calc_and_trap():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_rag_pipeline.db"
        create_database(db_path)
        repo = FinancialRepository(db_path)

        # Seed company, metric, report, facts, chunks
        c_id = repo.add_company(ticker="AAA", name_vi="Nhựa An Phát Xanh")
        m_rev = repo.add_metric(code="net_revenue", name_vi="Doanh thu thuần", name_en="Net revenue", aliases="Doanh thu thuần|Doanh thu")
        m_np = repo.add_metric(code="net_profit", name_vi="Lợi nhuận sau thuế", name_en="Net profit", aliases="Lợi nhuận sau thuế|LNST")
        
        r_id = repo.add_report(company_id=c_id, fiscal_year=2025, period_type="FY", currency="VND")

        repo.add_fact(
            report_id=r_id,
            line_item="Doanh thu thuần",
            column_role="current",
            value=4000.0,
            value_raw="4.000",
            metric_id=m_rev,
            page=7,
        )
        repo.add_fact(
            report_id=r_id,
            line_item="Lợi nhuận sau thuế",
            column_role="current",
            value=400.0,
            value_raw="400",
            metric_id=m_np,
            page=7,
        )

        repo.add_chunk(
            report_id=r_id,
            section="Thuyết minh",
            chunk_type="text",
            page_start=15,
            page_end=15,
            context_prefix="[AAA | FY/2025]",
            content="Chính sách ghi nhận doanh thu: Doanh thu bán hàng được ghi nhận khi đã chuyển giao rủi ro cho người mua.",
        )

        retriever = HybridRetriever(repo=repo)
        retriever.dense_retriever.index_path = Path(tmpdir) / "faiss.index"
        retriever.dense_retriever.chunk_ids_path = Path(tmpdir) / "chunk_ids.json"
        retriever.dense_retriever.build_index()

        llm = LLMBackend(backend_type="mock")
        engine = FinancialRAGEngine(repo=repo, llm=llm, retriever=retriever)

        # 1. Test Lookup
        lookup_res = engine.answer_question("Doanh thu thuần năm 2025 của AAA là bao nhiêu?")
        assert lookup_res.question_type == "lookup"
        assert len(lookup_res.citations) >= 1
        assert "4,000" in lookup_res.answer or "4000" in lookup_res.answer

        # 2. Test Calculation (Net Margin)
        calc_res = engine.answer_question("Biên lợi nhuận ròng của AAA năm 2025 là bao nhiêu phần trăm?")
        assert calc_res.question_type == "calc"
        assert "10.00%" in lookup_res.answer or "10.00%" in calc_res.answer

        # 3. Test Trap Question
        trap_res = engine.answer_question("Dự báo giá cổ phiếu AAA trong tháng tới sẽ là bao nhiêu?")
        assert trap_res.question_type == "trap"
        assert "từ chối" in trap_res.answer.lower() or "outside" in trap_res.answer.lower()
