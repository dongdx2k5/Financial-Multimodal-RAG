import sys
from pathlib import Path

import typer

from .config import settings
from .service import FinancialRAGService

app = typer.Typer(help="Bilingual (Vietnamese / English) financial Graph RAG")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _service() -> FinancialRAGService:
    return FinancialRAGService(settings)


@app.command()
def ingest(path: Path, ticker: str | None = None, year: int | None = None) -> None:
    """Ingest a PDF/TXT/HTML financial report."""
    typer.echo(_service().ingest(path, ticker=ticker, year=year))


@app.command()
def ask(
    question: str,
    top_k: int = 5,
    ticker: str | None = None,
    year: int | None = None,
) -> None:
    """Ask a question against the local hybrid index."""
    typer.echo(_service().ask(question, top_k=top_k, ticker=ticker, year=year).model_dump_json(indent=2))


@app.command()
def demo() -> None:
    """Ingest the bundled Vietnamese sample report."""
    path = Path(__file__).resolve().parents[2] / "samples" / "sample_report.txt"
    typer.echo(_service().ingest(path, ticker="DEMO", year=2024))
    typer.echo('Demo ready. Try: finrag ask "Doanh thu thuần năm 2024 là bao nhiêu?"')


@app.command(name="demo-en")
def demo_en() -> None:
    """Ingest the bundled English sample report (Apple Inc.)."""
    path = Path(__file__).resolve().parents[2] / "samples" / "sample_report_en.txt"
    typer.echo(_service().ingest(path, ticker="AAPL", year=2024))
    typer.echo('English demo ready. Try: finrag ask "What was Total Revenue in 2024?"')


@app.command(name="demo-all")
def demo_all() -> None:
    """Ingest both Vietnamese and English sample reports."""
    samples_dir = Path(__file__).resolve().parents[2] / "samples"
    svc = _service()
    typer.echo(svc.ingest(samples_dir / "sample_report.txt", ticker="DEMO", year=2024))
    typer.echo(svc.ingest(samples_dir / "sample_report_en.txt", ticker="AAPL", year=2024))
    typer.echo(
        "Bilingual demo ready.\n"
        '  Vietnamese: finrag ask "Doanh thu thuần năm 2024 là bao nhiêu?"\n'
        '  English:    finrag ask "What was Total Revenue in 2024?"\n'
        '  Cross-lang: finrag ask "Total Revenue của DEMO năm 2024?"'
    )


@app.command(name="docs")
def list_docs() -> None:
    """List all stored financial reports in the database."""
    docs = _service().list_documents()
    if not docs:
        typer.echo("Database is empty. Ingest some reports with 'finrag ingest <file>' or 'finrag demo'.")
        return
    typer.echo(f"Total documents: {len(docs)}")
    typer.echo("-" * 75)
    for doc in docs:
        typer.echo(
            f"ID: {doc['id']} | {doc['filename']} | Ticker: {doc['ticker'] or 'N/A'} | Year: {doc['year'] or 'N/A'} | Pages: {doc['pages_count']} | Chunks: {doc['chunks_count']} | Facts: {doc['facts_count']}"
        )


@app.command(name="delete-doc")
def delete_doc(document_id: str) -> None:
    """Delete a financial report from the database and graph."""
    success = _service().delete_document(document_id)
    if success:
        typer.echo(f"Successfully deleted document: {document_id}")
    else:
        typer.echo(f"Document not found: {document_id}")


@app.command()
def facts(
    ticker: str | None = None,
    year: int | None = None,
    metric: str | None = None,
    is_chart: bool | None = None,
    limit: int = 20,
) -> None:
    """Query structured financial facts directly from the database."""
    results = _service().get_facts(
        ticker=ticker, year=year, metric=metric, is_chart=is_chart, limit=limit
    )
    if not results:
        typer.echo("No facts found matching criteria.")
        return
    typer.echo(f"Found {len(results)} fact(s):")
    typer.echo("-" * 75)
    for f in results:
        typer.echo(
            f"[{f.ticker or 'N/A'}] {f.metric} ({f.period or 'N/A'}): {f.value} {f.unit or ''} (Page {f.page})"
        )


@app.command()
def stats() -> None:
    """Show database and knowledge graph statistics."""
    import json
    typer.echo(json.dumps(_service().get_stats(), indent=2))


if __name__ == "__main__":
    app()

