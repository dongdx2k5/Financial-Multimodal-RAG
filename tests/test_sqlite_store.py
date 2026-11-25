from pathlib import Path

from financial_rag.models import Chunk, Document, FinancialFact
from financial_rag.sqlite_store import FinancialSQLiteStore


def test_sqlite_store_lifecycle(tmp_path: Path):
    db_path = tmp_path / "test_financial.db"
    store = FinancialSQLiteStore(db_path)

    # 1. Test Document Storage
    doc = Document(id="doc_vnm_2024", source="vnm_2024.pdf", name="VNM BCTC 2024", ticker="VNM", year=2024, pages=[])
    store.save_document(doc, metadata={"status": "verified"})

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["id"] == "doc_vnm_2024"
    assert docs[0]["ticker"] == "VNM"

    # 2. Test Chunk Storage and FTS5 Search
    chunk1 = Chunk(
        id="chunk_1",
        document_id="doc_vnm_2024",
        source="vnm_2024.pdf",
        page=5,
        text="Doanh thu thuần bán hàng và cung cấp dịch vụ năm 2024 đạt 150000 tỷ VND",
        bbox=[10.0, 20.0, 500.0, 100.0],
    )
    chunk2 = Chunk(
        id="chunk_2",
        document_id="doc_vnm_2024",
        source="vnm_2024.pdf",
        page=6,
        text="Lợi nhuận sau thuế thu nhập doanh nghiệp đạt 12000 tỷ VND",
        bbox=[10.0, 110.0, 500.0, 200.0],
    )
    store.save_chunks([chunk1, chunk2])

    # FTS5 search
    search_res = store.search_chunks_fts("doanh thu", document_id="doc_vnm_2024")
    assert len(search_res) >= 1
    assert search_res[0]["chunk_id"] == "chunk_1"
    assert search_res[0]["page"] == 5

    search_res_profit = store.search_chunks_fts("lợi nhuận", document_id="doc_vnm_2024")
    assert len(search_res_profit) >= 1
    assert search_res_profit[0]["chunk_id"] == "chunk_2"

    # 3. Test Facts Storage and Retrieval
    fact = FinancialFact(
        id="fact_rev_2024",
        metric="Doanh thu thuần",
        metric_code="10",
        value=150000.0,
        raw_value="150.000",
        unit="tỷ VND",
        period=2024,
        period_label="2024",
        statement="income",
        document_id="doc_vnm_2024",
        chunk_id="chunk_1",
        source="vnm_2024.pdf",
        page=5,
        confidence=0.9,
    )
    store.save_facts([fact])

    facts = store.get_facts_by_code("doc_vnm_2024", "income", "10")
    assert len(facts) == 1
    assert facts[0].value == 150000.0
    assert facts[0].metric == "Doanh thu thuần"

    # 4. Test Human-in-the-loop fact update & cascading delete
    updated = store.update_fact("fact_rev_2024", value=155000.0, raw_value="155.000", quality="verified")
    assert updated is not None and updated.value == 155000.0 and updated.quality == "verified"

    deleted = store.delete_document("doc_vnm_2024")
    assert deleted is not None and deleted["id"] == "doc_vnm_2024"
    assert store.list_documents() == []
    assert store.load_chunks("doc_vnm_2024") == []
    assert store.load_facts("doc_vnm_2024") == []
    assert store.search_chunks_fts("doanh thu") == []
