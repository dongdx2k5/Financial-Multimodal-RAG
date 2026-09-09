import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from financial_rag.pipeline import ingest as ingest_path, ingest_dataset, query as query_rag

app = typer.Typer(help="Financial Multimodal RAG CLI (Text, Tables, Images)")
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to PDF, text file, or directory"),
    summarize: bool = typer.Option(
        True, "--summarize/--no-summarize", help="Run Multimodal Summarization (LLM for tables & vision for images)"
    ),
) -> None:
    """Extracts text, HTML tables, and images from document and indexes them into Multi-Vector Store."""
    console.print(f"[bold cyan]Ingesting multimodal content from:[/] {path} (summarize={summarize})")
    indexed = ingest_path(path, summarize_multimodal=summarize)
    console.print(f"[bold green]Successfully indexed {indexed} elements into Multi-Vector Store![/]")


@app.command("ingest-dataset")
def ingest_dataset_cmd(
    name: str = typer.Argument(..., help="Dataset name: finqa, vifintab, sec_edgar"),
    split: str = typer.Option("train", help="Dataset split: train, validation, test"),
    sample_size: int | None = typer.Option(None, help="Optional sample size limit"),
) -> None:
    """Ingest standard financial benchmark datasets."""
    indexed = ingest_dataset(name, split=split, sample_size=sample_size)
    console.print(
        f"[bold green]Successfully indexed {indexed} chunks from dataset '{name}' (split={split})[/bold green]"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the financial documents"),
    limit: int = typer.Option(6, "--limit", "-l", help="Number of retrieved context elements"),
) -> None:
    """Asks a question using the Multimodal QA Engine with text, tables, and images."""
    console.print(f"[bold yellow]Analyzing question with Multimodal RAG:[/] {question}\n")
    result = query_rag(question, limit=limit)

    # 1. Print Answer
    console.print(Panel(Markdown(result["answer"]), title="[bold green]AI Financial Analysis[/bold green]", expand=False))

    # 2. Print Retrieved Evidence Summary
    table = Table(title="Retrieved Context & Multimodal Evidence", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Source", style="white")
    table.add_column("Page", justify="right", width=6)
    table.add_column("Details", style="dim")

    for ctx in result.get("contexts", []):
        modality = ctx.get("modality", "text")
        src = str(ctx.get("metadata", {}).get("source", "unknown"))
        page = str(ctx.get("metadata", {}).get("page", "-"))
        summary = str(ctx.get("content", ""))[:80] + "..." if len(str(ctx.get("content", ""))) > 80 else str(ctx.get("content", ""))
        table.add_row(modality.upper(), Path(src).name, page, summary)

    console.print(table)

    images = result.get("retrieved_images", [])
    if images:
        console.print(f"\n[bold cyan]Retrieved {len(images)} Image(s)/Chart(s) analyzed by Vision LLM:[/bold cyan]")
        for img in images:
            img_path = img.get("metadata", {}).get("image_path", "In-memory")
            console.print(f" - Page {img.get('page')}: [underline]{img_path}[/underline]")


if __name__ == "__main__":
    app()
