import sys
from pathlib import Path

import typer

from .config import settings
from .service import FinancialRAGService

app = typer.Typer(help="Vietnamese financial Graph RAG MVP")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _service() -> FinancialRAGService:
    return FinancialRAGService(settings)


@app.command()
def ingest(path: Path, ticker: str | None = None, year: int | None = None) -> None:
    """Ingest a PDF/TXT/HTML financial report."""
    typer.echo(_service().ingest(path, ticker=ticker, year=year))


@app.command()
def ask(question: str, top_k: int = 5) -> None:
    """Ask a question against the local hybrid index."""
    typer.echo(_service().ask(question, top_k).model_dump_json(indent=2))


@app.command()
def demo() -> None:
    """Ingest the bundled sample report."""
    path = Path(__file__).resolve().parents[2] / "samples" / "sample_report.txt"
    typer.echo(_service().ingest(path, ticker="DEMO", year=2024))
    typer.echo('Demo ready. Try: finrag ask "Doanh thu thuần năm 2024 là bao nhiêu?"')


if __name__ == "__main__":
    app()
