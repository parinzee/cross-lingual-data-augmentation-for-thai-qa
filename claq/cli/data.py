import os
import typer
from ..data import load_unprocessed_qa_dataset
from rich import print
from pathlib import Path

app = typer.Typer()

@app.command()
def download(force: bool = typer.Option(False)):
    path = Path("data/original_dataset.parquet")
    if os.path.exists(path) and not force:
        print(f"[red]{path} already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(1)
    elif force:
        print(f"[red]Overwriting {path}...[/red]")
    
    # Make folder if it doesn't exist
    os.makedirs(path.parent, exist_ok=True)

    # Download dataset
    loaded = load_unprocessed_qa_dataset()
    print(f"[green]Saving to {path}...[/green]")
    loaded.to_parquet(path, index=False)