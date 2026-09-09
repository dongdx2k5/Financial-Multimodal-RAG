import tempfile
from pathlib import Path
from financial_rag.ingestion.document_loader import DocumentLoader, _table_to_html
from financial_rag.ingestion.models import Modality


def test_table_to_html_formatting() -> None:
    grid = [
        ["Metric", "2022", "2023"],
        ["Revenue", "$10M", "$15M"],
        ["Net Margin", "12%", "14%"],
    ]
    html_output = _table_to_html(grid)
    assert "<table>" in html_output
    assert "<thead><tr><th>Metric</th><th>2022</th><th>2023</th></tr></thead>" in html_output
    assert "<tbody><tr><td>Revenue</td><td>$10M</td><td>$15M</td></tr>" in html_output
    assert "</table>" in html_output


def test_document_loader_txt_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "notes.txt"
        test_file.write_text("Quarterly financial notes", encoding="utf-8")

        loader = DocumentLoader()
        elements = loader.load_file(test_file)

        assert len(elements) == 1
        assert elements[0].modality == Modality.TEXT
        assert elements[0].content == "Quarterly financial notes"
        assert elements[0].raw_content == "Quarterly financial notes"
