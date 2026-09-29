from financial_rag.graph import KnowledgeGraph
from financial_rag.models import Chunk
from financial_rag.retrieval import HybridRetriever


def test_cashflow_alias_avoids_contents(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.json")
    retriever = HybridRetriever(tmp_path / "chunks.json", graph)
    retriever.upsert(
        [
            Chunk(
                id="toc",
                document_id="a",
                source="a.pdf",
                page=2,
                text="NỘI DUNG TRANG dòng tiền kinh doanh thuyết minh",
            ),
            Chunk(
                id="cash",
                document_id="a",
                source="a.pdf",
                page=11,
                text="Lưu chuyển tiền thuần từ hoạt động kinh doanh: (86.395.780.249) VND",
            ),
            Chunk(
                id="policy",
                document_id="a",
                source="a.pdf",
                page=24,
                text="Chính sách ghi nhận doanh thu kinh doanh",
            ),
        ]
    )
    result = retriever.search("Tại sao dòng tiền kinh doanh âm?", document_id="a")
    assert result[0].chunk_id == "cash"
    assert all(e.chunk_id != "toc" for e in result)


def test_policy_prefers_explanation_over_numeric_row(tmp_path):
    graph = KnowledgeGraph(tmp_path / "graph.json")
    retriever = HybridRetriever(tmp_path / "chunks.json", graph)
    retriever.upsert(
        [
            Chunk(
                id="row",
                document_id="a",
                source="a.pdf",
                page=10,
                text="Doanh thu bán hàng",
                financial_rows=[{"value": 100}],
            ),
            Chunk(
                id="policy",
                document_id="a",
                source="a.pdf",
                page=24,
                text="Doanh thu bán hàng được ghi nhận khi chuyển giao rủi ro và lợi ích.",
            ),
        ]
    )
    assert retriever.search("Điều kiện ghi nhận doanh thu bán hàng", document_id="a")[0].page == 24
