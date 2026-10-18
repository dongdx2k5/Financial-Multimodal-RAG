"""Command line interface for Financial RAG."""

import os
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from financial_rag.config import settings
from financial_rag.db.repository import FinancialRepository
from financial_rag.db.seed import seed_database
from financial_rag.evaluation.evaluator import SystemEvaluator
from financial_rag.extractor.fact_extractor import FactExtractor
from financial_rag.extractor.validator import FinancialDataValidator
from financial_rag.parser.chunker import FinancialChunker
from financial_rag.parser.pdf_parser import FinancialPDFParser
from financial_rag.reasoning.engine import FinancialRAGEngine
from financial_rag.retrieval.dense import DenseRetriever

app = typer.Typer(help="Financial Multimodal RAG CLI")
console = Console()


@app.command()
def seed():
    """Initialize database tables, seed metrics dictionary and manifest metadata."""
    console.print("[bold green]Seeding database...[/bold green]")
    seed_database()
    console.print("[bold green]Database schema and dictionary initialized successfully![/bold green]")


@app.command()
def ingest(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Specific company ticker to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already parsed"),
):
    """Parse raw financial reports, extract facts into SQLite, chunk texts, and index."""
    db_path = settings.resolve_path(settings.database_path)
    repo = FinancialRepository(db_path)
    parser = FinancialPDFParser()
    chunker = FinancialChunker()
    extractor = FactExtractor(repo)
    validator = FinancialDataValidator(repo)

    reports = repo.find_reports(ticker=ticker)
    if not reports:
        console.print("[yellow]No reports found in manifest. Running seed first...[/yellow]")
        seed_database()
        reports = repo.find_reports(ticker=ticker)

    console.print(f"[bold cyan]Found {len(reports)} report(s) to process.[/bold cyan]")

    total_facts = 0
    total_chunks = 0

    for r in reports:
        rep_id = r["report_id"]
        t = r["ticker"]
        fy = r["fiscal_year"]
        pt = r["period_type"]
        file_path = r.get("file_path")

        if not file_path:
            console.print(f"[yellow]Skipping {t} ({pt}/{fy}): No file path provided.[/yellow]")
            continue

        resolved_file = settings.resolve_path(Path(file_path))
        if not resolved_file.exists():
            console.print(f"[red]File not found: {resolved_file}[/red]")
            continue

        console.print(f"[green]Parsing {t} ({pt}/{fy}) from {resolved_file.name}...[/green]")
        try:
            parse_res = parser.parse(resolved_file)
            
            # Save parsed artifacts
            out_dir = settings.resolve_path(settings.data_dir / "parsed" / f"{t}_{fy}_{pt}")
            parser.save_parsed_artifacts(parse_res, out_dir)

            # 1. Chunk document
            chunks = chunker.chunk_document(
                parse_result=parse_res,
                report_id=rep_id,
                ticker=t,
                period_type=pt,
                fiscal_year=fy,
                scope=r.get("scope", "consolidated"),
                currency=r.get("currency", "VND"),
                unit_scale=int(r.get("unit_scale", 1)),
            )
            chunk_dicts = [c.to_dict() for c in chunks]
            added_chunk_ids = repo.add_chunks_batch(chunk_dicts)
            total_chunks += len(added_chunk_ids)

            # 2. Extract facts
            facts = extractor.extract_from_parsed_document(
                parse_result=parse_res,
                report_id=rep_id,
                fiscal_year=fy,
                unit_scale=int(r.get("unit_scale", 1)),
            )
            added_facts = repo.add_facts_batch(facts)
            total_facts += added_facts

            # 3. Validate
            val_res = validator.validate_report(report_id=rep_id)
            if not val_res.is_valid:
                console.print(f"[yellow]  Validation warnings for {t}: {val_res.errors}[/yellow]")
            else:
                console.print(f"[green]  Validation passed: Balance sheet equation holds.[/green]")

            console.print(f"  -> Added {len(added_chunk_ids)} chunks and {added_facts} structured facts.")

        except Exception as e:
            console.print(f"[bold red]Error ingesting report {t}: {e}[/bold red]")

    # Build dense vector index
    console.print("[bold cyan]Building dense vector index...[/bold cyan]")
    dense_retriever = DenseRetriever(repo)
    indexed_count = dense_retriever.build_index()
    console.print(f"[bold green]Ingestion complete! Indexed {indexed_count} chunks, {total_facts} total facts.[/bold green]")


@app.command("ingest-file")
def ingest_file_cmd(
    file_path: str = typer.Argument(..., help="Path to financial report PDF or text file"),
    ticker: str = typer.Option(..., "--ticker", "-t", help="Stock ticker (e.g. FPT, VNM, VCB)"),
    fiscal_year: int = typer.Option(..., "--year", "-y", help="Fiscal year (e.g. 2025)"),
    period_type: str = typer.Option("FY", "--period", "-p", help="Period type: FY, H1, Q1, Q2, Q3, Q4"),
    unit_scale: int = typer.Option(1, "--scale", "-s", help="Unit scale: 1, 1000, 1000000 (triệu), 1000000000 (tỷ)"),
    currency: str = typer.Option("VND", "--currency", "-c", help="Currency: VND or USD"),
    scope: str = typer.Option("consolidated", "--scope", help="consolidated (hợp nhất) or separate (riêng lẻ)"),
    company_name: Optional[str] = typer.Option(None, "--company", help="Company name (optional)"),
):
    """Ingest a single PDF or text financial report directly without needing to create subfolders."""
    from financial_rag.parser.ingest_pipeline import ingest_single_file

    file_path = file_path.strip("\"'")
    console.print(f"[bold cyan]Ingesting file: {file_path} for {ticker} ({period_type}/{fiscal_year})...[/bold cyan]")
    try:
        res = ingest_single_file(
            file_path=file_path,
            ticker=ticker,
            fiscal_year=fiscal_year,
            period_type=period_type,
            unit_scale=unit_scale,
            currency=currency,
            scope=scope,
            company_name_vi=company_name,
        )
        console.print(f"[bold green]Ingestion successful for {res['ticker']}![/bold green]")
        console.print(f"  - Pages parsed: [cyan]{res['pages_parsed']}[/cyan]")
        console.print(f"  - Chunks indexed: [cyan]{res['chunks_indexed']}[/cyan]")
        console.print(f"  - Facts extracted: [cyan]{res['facts_extracted']}[/cyan]")
        if res['is_valid']:
            console.print("  - [green]Accounting equation holds (Assets = Liabilities + Equity)[/green]")
        else:
            console.print(f"  - [yellow]Validation notice: {res['validation_errors']}[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Failed to ingest file: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def ask(question: str):
    """Ask a question to the Financial RAG pipeline in terminal."""
    db_path = settings.resolve_path(settings.database_path)
    repo = FinancialRepository(db_path)
    engine = FinancialRAGEngine(repo=repo)

    console.print(f"[bold yellow]Question:[/bold yellow] {question}\n")
    response = engine.answer_question(question)

    console.print("[bold green]Answer:[/bold green]")
    console.print(response.answer)

    if response.citations:
        console.print("\n[bold cyan]Citations:[/bold cyan]")
        for c in response.citations:
            console.print(f"  - {c}")

    if response.verification and response.verification.get("warning"):
        console.print(f"\n[bold red]{response.verification['warning']}[/bold red]")


@app.command()
def evaluate():
    """Run evaluation benchmark across all test questions."""
    db_path = settings.resolve_path(settings.database_path)
    repo = FinancialRepository(db_path)
    engine = FinancialRAGEngine(repo=repo)
    evaluator = SystemEvaluator(engine=engine, repo=repo)

    console.print("[bold cyan]Running evaluation benchmark...[/bold cyan]")
    result = evaluator.evaluate_all()

    table = Table(title="Financial RAG Evaluation Benchmark Results")
    table.add_column("Metric / Question Type", style="cyan")
    table.add_column("Score / Accuracy", style="green")

    table.add_row("Total Questions", str(result.total_questions))
    table.add_row("Lookup Accuracy", f"{result.lookup_accuracy}%")
    table.add_row("Calculation Accuracy", f"{result.calc_accuracy}%")
    table.add_row("Text Explanation Accuracy", f"{result.text_accuracy}%")
    table.add_row("Comparison Accuracy", f"{result.compare_accuracy}%")
    table.add_row("Trap Refusal Rate", f"{result.trap_refusal_rate}%")
    table.add_row("Citation Correctness", f"{result.citation_accuracy}%")
    table.add_row("[bold]Overall Benchmark Score[/bold]", f"[bold]{result.overall_score}%[/bold]")

    console.print(table)


@app.command()
def serve(
    port: int = typer.Option(7860, "--port", "-p", help="Port for Gradio UI"),
    share: bool = typer.Option(False, "--share", help="Create public shareable link"),
):
    """Launch interactive Gradio Web UI."""
    try:
        from financial_rag.ui.app import create_gradio_app
    except ImportError as e:
        console.print(f"[bold red]Gradio is required to run the Web UI: {e}[/bold red]")
        raise typer.Exit(1)

    db_path = settings.resolve_path(settings.database_path)
    repo = FinancialRepository(db_path)
    engine = FinancialRAGEngine(repo=repo)

    demo = create_gradio_app(engine=engine, repo=repo)
    console.print(f"[bold green]Starting Gradio server on port {port}...[/bold green]")
    demo.launch(server_port=port, share=share)


@app.command()
def doctor():
    """Check system environment, database status, and Colab GPU readiness."""
    console.print("[bold]System Diagnostics (Doctor):[/bold]\n")

    # DB Check
    db_path = settings.resolve_path(settings.database_path)
    if db_path.exists():
        repo = FinancialRepository(db_path)
        comps = repo.get_all_companies()
        facts = repo.query_facts()
        chunks = repo.get_all_chunks()
        console.print(f"[bold green][OK][/bold green] SQLite Database: {db_path} ({len(comps)} companies, {len(facts)} facts, {len(chunks)} chunks)")
    else:
        console.print(f"[bold red][FAIL][/bold red] SQLite Database: Not found at {db_path}. Run `finrag seed`.")

    # GPU / PyTorch Check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
        console.print(f"[bold green][OK][/bold green] PyTorch: {torch.__version__} (CUDA: {cuda_avail}, GPU: {gpu_name})")
    except ImportError:
        console.print("[yellow][WARN][/yellow] PyTorch not installed (running in lightweight CPU mode).")

    # Transformers check
    try:
        import transformers
        console.print(f"[bold green][OK][/bold green] Transformers: {transformers.__version__}")
    except ImportError:
        console.print("[yellow][WARN][/yellow] Transformers not installed (using Mock/OpenAI backend).")


if __name__ == "__main__":
    app()
