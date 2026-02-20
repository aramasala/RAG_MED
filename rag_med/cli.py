"""Command-line interface for RAG_MED."""

import logging
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from . import __version__
from .pdf_cleaner import clean_pdf
from .qa_generator import generate_qa_from_pdf

app = typer.Typer(
    name="rag-med",
    help="RAG_MED - Medical RAG system for PDF processing and QA generation",
    add_completion=False,
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@app.command()
def version() -> None:
    """Show version information."""
    rprint(f"[bold green]RAG_MED[/bold green] v{__version__}")


@app.command()
def clean(
    input_path: Path = typer.Argument(..., help="Input PDF file or directory"),
    output_path: Path | None = typer.Option(
        None, "--output", "-o", help="Output path (default: {input}_cleaned.pdf)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Clean PDF files by removing unnecessary sections.

    Example:
        rag-med clean document.pdf
        rag-med clean /path/to/pdfs/ --output /path/to/output/
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 50)
    logger.info("PDF Cleaner - запуск")
    logger.info("=" * 50)

    pdf_files: list[Path] = []

    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = list(input_path.rglob("*.pdf"))
        if not pdf_files:
            console.print("[red]В папке PDF файлы не найдены[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Неверный путь: {input_path}[/red]")
        raise typer.Exit(1)

    success_count = 0

    for pdf_file in pdf_files:
        if output_path and output_path.is_dir():
            output_file = output_path / f"{pdf_file.stem}_cleaned.pdf"
        elif output_path:
            output_file = output_path
        else:
            output_file = pdf_file.with_name(f"{pdf_file.stem}_cleaned.pdf")

        if clean_pdf(pdf_file, output_file):
            success_count += 1
            console.print(f"[green] Обработан: {pdf_file} -> {output_file}[/green]")
        else:
            console.print(f"[red] Ошибка при обработке: {pdf_file}[/red]")

    logger.info("=" * 50)
    logger.info(f"Готово. Обработано файлов: {success_count} из {len(pdf_files)}")
    logger.info("=" * 50)


def _build_results_table(results: list, valueai_eval: bool) -> None:
    """Render results table and print to console."""
    table = Table(title=" QA Generation Results", show_header=True)
    table.add_column("Chunk", style="cyan", width=10)
    table.add_column("Question", style="green", width=40)
    table.add_column("Answer", style="white", width=40)
    if valueai_eval:
        table.add_column("ValueAI", style="magenta", width=40)
        table.add_column("Score", style="yellow", width=8)
    for r in results:
        valueai_text = ""
        score_text = ""
        if valueai_eval:
            valueai_text = r.valueai_answer or ""
            metrics = r.evaluation_metrics or {}
            if "primary_score" in metrics:
                score_text = f"{metrics['primary_score']:.3f}"
            elif "error" in metrics:
                score_text = "ERR"
        row = [
            str(r.chunk_index),
            r.question[:60] + "..." if len(r.question) > 60 else r.question,
            r.answer[:60] + "..." if len(r.answer) > 60 else r.answer,
        ]
        if valueai_eval:
            row.append(valueai_text[:60] + "..." if len(valueai_text) > 60 else valueai_text)
            row.append(score_text)
        table.add_row(*row)
    console.print(table)


@app.command()
def generate(
    pdf_path: Path = typer.Argument(..., help="Input PDF file"),
    output_file: str = typer.Option("qa_result.json", "--output", "-o", help="Output JSON file"),
    num_questions: int | None = typer.Option(
        None, "--num-questions", "-n", help="Number of questions to generate"
    ),
    valueai_eval: bool = typer.Option(False, "--valueai-eval", help="Evaluate with ValueAI"),
    eval_summary: str | None = typer.Option(None, "--eval-summary", help="Evaluation summary file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Generate QA pairs from PDF file.

    Example:
        rag-med generate document.pdf
        rag-med generate document.pdf --output results.json
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Convert string paths to Path objects
        output_path = Path(output_file) if output_file else None
        summary_path = Path(eval_summary) if eval_summary else None

        results = generate_qa_from_pdf(
            pdf_path,
            output_path,
            num_questions=num_questions,
            evaluate_with_valueai=valueai_eval,
            summary_file=summary_path,
        )
        _build_results_table(results, valueai_eval)
        console.print(f"\n[green] Results saved to: {output_file}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red] Error: {e}[/red]")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red] Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red] Unexpected error: {e}[/red]")
        logger.exception("Unexpected error")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
