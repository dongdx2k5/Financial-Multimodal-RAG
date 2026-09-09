from pathlib import Path
from financial_rag.ingestion.models import DocumentElement, Modality
from financial_rag.retrieval.chunking import chunk_elements


def test_chunk_elements_keeps_source_metadata() -> None:
    element = DocumentElement("Revenue grew while margin declined." * 50, "text", Path("report.pdf"), page=3)
    chunks = chunk_elements([element], chunk_size=80, overlap=10)
    assert chunks
    assert chunks[0].source_path == Path("report.pdf")
    assert chunks[0].page == 3
    assert chunks[0].modality == "text"


def test_chunk_elements_handles_images() -> None:
    element = DocumentElement(
        content="Chart showing Q3 revenue by sector",
        modality=Modality.IMAGE,
        source_path=Path("chart.png"),
        page=2,
        raw_content="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        summary="Bar chart of Q3 revenue",
        metadata={"image_name": "chart.png"},
    )
    chunks = chunk_elements([element])
    assert len(chunks) == 1
    assert chunks[0].modality == "image"
    assert chunks[0].content == "Bar chart of Q3 revenue"
    assert chunks[0].raw_content.startswith("iVBOR")


def test_chunk_elements_handles_tables() -> None:
    table_html = "<table><thead><tr><th>Year</th><th>Revenue</th></tr></thead><tbody><tr><td>2023</td><td>$100M</td></tr></tbody></table>"
    element = DocumentElement(
        content=table_html,
        modality=Modality.TABLE,
        source_path=Path("table.pdf"),
        page=5,
        raw_content=table_html,
        summary="Summary of 2023 revenue table ($100M)",
    )
    chunks = chunk_elements([element])
    assert len(chunks) == 1
    assert chunks[0].modality == "table"
    assert chunks[0].content == "Summary of 2023 revenue table ($100M)"
    assert "<table>" in chunks[0].raw_content
