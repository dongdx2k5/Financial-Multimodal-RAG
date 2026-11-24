from pathlib import Path

import pymupdf

from financial_rag.chart_reader import ChartReader
from financial_rag.graph import KnowledgeGraph
from financial_rag.models import Document
from scripts.create_sample_pdf import generate_sample_pdf


def test_chart_reader_extraction(tmp_path: Path):
    pdf_path = tmp_path / "chart_report.pdf"
    generate_sample_pdf(pdf_path)

    doc_fitz = pymupdf.open(str(pdf_path))
    page = doc_fitz[0]

    # Chart box from generate_sample_pdf: Rect(50, 360, 545, 580)
    chart_boxes = [[50.0, 360.0, 545.0, 580.0]]
    document = Document(id="doc_chart_demo", source="chart_report.pdf", ticker="DEMO", year=2024, pages=[])

    reader = ChartReader()
    results = reader.extract_from_page(page, page_num=1, chart_boxes=chart_boxes, document=document)

    assert len(results) == 1
    res = results[0]
    assert "Biểu đồ" in res.title or "doanh thu" in res.title.lower()
    assert res.chunk is not None
    assert res.chunk.extraction_method == "chart"

    # Verify Knowledge Graph integration
    graph_path = tmp_path / "graph.json"
    kg = KnowledgeGraph(graph_path)
    kg.add_document(document, [res.chunk], res.facts)

    assert document.id in kg.graph
    assert res.chunk.id in kg.graph
    # Verify edge MENTIONS exists between Document and Chart Chunk
    assert kg.graph.has_edge(document.id, res.chunk.id)
