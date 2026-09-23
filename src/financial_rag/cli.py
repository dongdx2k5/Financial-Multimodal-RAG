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

from financial_rag.pipeline import (
    ingest as ingest_path,
    ingest_dataset,
    query as query_rag,
    build_graph as build_graph_pipeline,
    graph_stats as graph_stats_pipeline,
)

app = typer.Typer(help="Financial GraphRAG CLI — Knowledge Graph + Vector hybrid RAG")
console = Console()


# ---------------------------------------------------------------------------
# ingest — PDF / directory ingestion
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to PDF, text file, or directory"),
    summarize: bool = typer.Option(
        True, "--summarize/--no-summarize",
        help="Run Multimodal Summarization (LLM for tables & vision for images)",
    ),
    max_pages: int | None = typer.Option(
        None, "--max-pages", "-p", help="Giới hạn số trang PDF cần xử lý",
    ),
) -> None:
    """Extract text, HTML tables, and images from a PDF and index into Vector Store."""
    console.print(f"[bold cyan]Ingesting:[/] {path} (summarize={summarize}, max_pages={max_pages})")
    indexed = ingest_path(path, summarize_multimodal=summarize, max_pages=max_pages)
    console.print(f"[bold green]Indexed {indexed} elements into Multi-Vector Store.[/]")


# ---------------------------------------------------------------------------
# ingest-dataset — HuggingFace benchmark datasets
# ---------------------------------------------------------------------------

@app.command("ingest-dataset")
def ingest_dataset_cmd(
    name: str = typer.Argument(..., help="Dataset name: finqa | vifintab | sec_edgar"),
    split: str = typer.Option("train", help="Dataset split: train | validation | test"),
    sample_size: int | None = typer.Option(None, help="Limit number of records"),
    build_kg: bool = typer.Option(True, "--build-kg/--no-build-kg", help="Also build/update Knowledge Graph"),
    llm_extract: bool = typer.Option(
        False, "--llm-extract/--no-llm-extract",
        help="Use LLM entity extraction for richer graph triples (slow, costs API calls)",
    ),
) -> None:
    """Ingest a benchmark dataset into Vector Store + Knowledge Graph."""
    console.print(f"[bold cyan]Ingesting dataset:[/] {name} (split={split}, sample={sample_size}, build_kg={build_kg})")
    result = ingest_dataset(
        name, split=split, sample_size=sample_size,
        build_kg=build_kg, use_llm_extraction=llm_extract,
    )
    if isinstance(result, dict):
        console.print(f"[bold green]✓ Vector chunks:[/]      {result.get('vector_chunks', 0)}")
        if build_kg:
            console.print(f"[bold green]✓ Graph nodes:[/]        {result.get('graph_nodes', 0)}")
            console.print(f"[bold green]✓ Graph edges:[/]        {result.get('graph_edges', 0)}")
            console.print(f"[bold green]✓ SAME_AS edges (VI↔EN):[/] {result.get('same_as_edges', 0)}")
    else:
        console.print(f"[bold green]Successfully indexed {result} chunks from '{name}'[/bold green]")


# ---------------------------------------------------------------------------
# build-graph — build Knowledge Graph only (no vector store update)
# ---------------------------------------------------------------------------

@app.command("build-graph")
def build_graph_cmd(
    dataset: str = typer.Argument("both", help="Dataset: finqa | vifintab | both"),
    split: str = typer.Option("train", help="Dataset split"),
    sample_size: int | None = typer.Option(None, help="Limit records per dataset"),
    llm_extract: bool = typer.Option(
        False, "--llm-extract/--no-llm-extract",
        help="Use LLM entity extraction (slow but produces richer triples)",
    ),
) -> None:
    """Build the Knowledge Graph from FinQA/ViFinTab (does NOT update Vector Store)."""
    console.print(f"[bold cyan]Building Knowledge Graph from:[/] {dataset} | split={split} | sample={sample_size}")
    stats = build_graph_pipeline(
        dataset=dataset,
        split=split,
        sample_size=sample_size,
        use_llm_extraction=llm_extract,
        link_bilingual=True,
    )
    t = Table(title="Knowledge Graph Statistics", show_header=True, header_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right", style="bold white")
    for k, v in stats.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                t.add_row(f"  {sub_k}", str(sub_v))
        else:
            t.add_row(k, str(v))
    console.print(t)


# ---------------------------------------------------------------------------
# graph-stats — show Knowledge Graph statistics
# ---------------------------------------------------------------------------

@app.command("graph-stats")
def graph_stats_cmd() -> None:
    """Show statistics about the persisted Knowledge Graph."""
    stats = graph_stats_pipeline()
    if "error" in stats:
        console.print(f"[bold red]{stats['error']}[/bold red]")
        raise typer.Exit(code=1)
    t = Table(title="Knowledge Graph Stats", show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right", style="bold white")
    for k, v in stats.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                t.add_row(f"  {sub_k}", str(sub_v))
        else:
            t.add_row(k, str(v))
    console.print(t)


# ---------------------------------------------------------------------------
# ask — GraphRAG question answering
# ---------------------------------------------------------------------------

@app.command()
def ask(
    question: str = typer.Argument(..., help="Financial question to answer"),
    limit: int = typer.Option(6, "--limit", "-l", help="Number of vector context chunks"),
    use_pot: bool = typer.Option(
        False, "--pot/--no-pot",
        help="Force Program-of-Thought (LLM generates Python code for arithmetic)",
    ),
    no_graph: bool = typer.Option(
        False, "--no-graph",
        help="Disable Knowledge Graph retrieval (vector-only mode)",
    ),
) -> None:
    """Ask a question using GraphRAG — hybrid Knowledge Graph + Vector retrieval + PoT."""
    mode = "Vector-only" if no_graph else "GraphRAG (KG + Vector)"
    console.print(f"[bold yellow]Mode:[/] {mode}")
    console.print(f"[bold yellow]Question:[/] {question}\n")

    result = query_rag(question, limit=limit, use_pot=use_pot, use_graph=not no_graph)

    # ── 1. Answer panel ───────────────────────────────────────────────
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]AI Financial Analysis[/bold green]",
        expand=False,
    ))

    # ── 2. Retrieval mode + Graph context summary ─────────────────────
    retrieval_mode = result.get("retrieval_mode", "unknown")
    console.print(f"[dim]Retrieval mode: {retrieval_mode}[/dim]")

    graph_info = result.get("graph_retrieval", {})
    if graph_info and graph_info.get("node_count", 0) > 0:
        n_nodes = graph_info.get("node_count", 0)
        n_edges = graph_info.get("edge_count", 0)
        seeds   = graph_info.get("seed_nodes", [])
        times   = graph_info.get("time_anchors", [])
        console.print(
            f"[bold cyan]Graph context:[/] {n_nodes} nodes · {n_edges} edges · "
            f"{len(seeds)} seed entities · time anchors: {times or 'none'}"
        )

    # ── 3. PoT results ────────────────────────────────────────────────
    if result.get("generated_code"):
        console.print(Panel(
            result["generated_code"],
            title="[bold magenta]Generated Python Code (Program-of-Thought)[/bold magenta]",
            expand=False,
        ))
        computed = result.get("computed_answer")
        if computed is not None:
            console.print(f"[bold cyan]Computed Answer:[/] {computed}")
        if not result.get("pot_success", True):
            console.print("[bold red]⚠ Code execution failed — answer generated by LLM text only.[/bold red]")
        console.print()

    # ── 4. Evidence table ─────────────────────────────────────────────
    evidence_table = Table(
        title="Retrieved Context & Evidence",
        show_header=True,
        header_style="bold magenta",
    )
    evidence_table.add_column("Type", style="cyan", width=16)
    evidence_table.add_column("Source", style="white")
    evidence_table.add_column("Page", justify="right", width=6)
    evidence_table.add_column("Preview", style="dim")

    for ctx in result.get("contexts", []):
        modality = ctx.get("modality", "text")
        src      = str(ctx.get("metadata", {}).get("source", "unknown"))
        page     = str(ctx.get("metadata", {}).get("page", "-"))
        text     = str(ctx.get("content", ""))
        preview  = text[:80] + "..." if len(text) > 80 else text
        evidence_table.add_row(modality.upper(), Path(src).name, page, preview)

    console.print(evidence_table)

    images = result.get("retrieved_images", [])
    if images:
        console.print(f"\n[bold cyan]Retrieved {len(images)} Image(s) analyzed by Vision LLM:[/bold cyan]")
        for img in images:
            console.print(f" - Page {img.get('page')}: {img.get('source', 'unknown')}")


if __name__ == "__main__":
    app()
