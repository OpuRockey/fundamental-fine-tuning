from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from .pipeline import prepare_dataset

app = typer.Typer(
    name="llm-data",
    help="Validate and prepare JSONL datasets for LLM supervised fine-tuning.",
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    """Prepare JSONL datasets for LLM supervised fine-tuning."""


@app.command()
def prepare(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("data/processed"), "--output-dir"),
    report_dir: Path = typer.Option(Path("data/reports"), "--report-dir"),
    model_id: str | None = typer.Option(
        os.environ.get("MODEL_ID"),
        "--model-id",
        help="Hugging Face model ID or local tokenizer directory with a chat template.",
    ),
    train_ratio: float = typer.Option(0.8, min=0.0, max=1.0),
    validation_ratio: float = typer.Option(0.1, min=0.0, max=1.0),
    test_ratio: float = typer.Option(0.1, min=0.0, max=1.0),
    seed: int = typer.Option(42),
    max_records: int = typer.Option(1_000_000, min=1),
    max_content_chars: int = typer.Option(100_000, min=1),
) -> None:
    """Prepare JSONL records, detect exact duplicates/conflicts, split, and optionally format."""
    try:
        report = prepare_dataset(
            input_path=input_path,
            output_dir=output_dir,
            report_dir=report_dir,
            model_id=model_id,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            max_records=max_records,
            max_content_chars=max_content_chars,
        )
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title="Dataset preparation complete")
    table.add_column("Metric")
    table.add_column("Count", justify="right")
    for key, value in report["counts"].items():
        if isinstance(value, (int, float)):
            table.add_row(key, str(value))
    for split_name, count in report["counts"]["splits"].items():
        table.add_row(f"split.{split_name}", str(count))
    print(table)
    print(f"[green]Output:[/green] {output_dir.resolve()}")
    print(f"[green]Reports:[/green] {report_dir.resolve()}")


if __name__ == "__main__":
    app()
