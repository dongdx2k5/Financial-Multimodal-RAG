from pathlib import Path

import pymupdf

from financial_rag.chart_reader import ChartReader
from financial_rag.extraction import FinancialFactExtractor
from financial_rag.graph import KnowledgeGraph
from financial_rag.ingestion import Chunker
from financial_rag.models import Document, PageContent
from scripts.create_sample_pdf import generate_sample_pdf


def test_chart_reader_extraction(tmp_path: Path):
    pdf_path = tmp_path / "chart_report.pdf"
    generate_sample_pdf(pdf_path)

    with pymupdf.open(str(pdf_path)) as doc_fitz:
        page = doc_fitz[0]
        chart_boxes = [[50.0, 360.0, 545.0, 580.0]]
        document = Document(
            id="doc_chart_demo", source="chart_report.pdf", ticker="DEMO", year=2024, pages=[]
        )
        results = ChartReader().extract_from_page(
            page, page_num=1, chart_boxes=chart_boxes, document=document
        )

    assert len(results) == 1
    res = results[0]
    assert "Biểu đồ" in res.title or "doanh thu" in res.title.lower()
    assert res.chunk is not None
    assert res.chunk.extraction_method == "chart"

    kg = KnowledgeGraph(tmp_path / "graph.json")
    kg.add_document(document, [res.chunk], res.facts)
    assert kg.graph.has_edge(document.id, res.chunk.id)


def test_chart_reader_with_numeric_labels_never_marks_consistent(tmp_path: Path):
    pdf_path = tmp_path / "labeled_chart.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=400)
        page.insert_text((50, 40), "Biểu đồ doanh thu (tỷ VND)")
        page.draw_rect(pymupdf.Rect(50, 60, 550, 300))
        page.insert_text((120, 150), "120,5")
        page.insert_text((120, 280), "2023")
        page.insert_text((320, 120), "155,0")
        page.insert_text((320, 280), "2024")
        doc.save(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        document = Document(id="doc_num", source=str(pdf_path), scope="consolidated", pages=[])
        res = ChartReader().extract_from_page(doc[0], 1, [[50, 60, 550, 300]], document)[0]

    assert [(dp.period, dp.value) for dp in res.data_points] == [(2023, 120.5), (2024, 155.0)]
    assert all(f.quality == "unverified" for f in res.facts)

    # Verify chunk -> extractor preserves document_id, scope, and bbox
    page_content = PageContent(page=1, text="", chart_results=[res.model_dump()])
    chunks = Chunker().split(Document(id="doc_num", source="c.pdf", scope="consolidated", pages=[page_content]))
    chart_chunks = [c for c in chunks if c.extraction_method == "chart"]
    extracted = FinancialFactExtractor().extract(chart_chunks[0])
    assert len(extracted) == 2
    assert all(f.document_id == "doc_num" and f.scope == "consolidated" and f.quality == "unverified" for f in extracted)
