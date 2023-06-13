import os
import typer
from ..data import load_unprocessed_qa_dataset
from rich import print

app = typer.Typer()

@app.command()
def download(path: str = typer.Option("data/original_dataset.csv"), force: bool = typer.Option(False)):
    if os.path.exists(path) and not force:
        print(f"[red]{path} already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(1)
    else:
        print(f"[red]Overwriting {path}...[/red]")
    loaded = load_unprocessed_qa_dataset()
    print(f"[green]Saving to {path}...[/green]")
    loaded.to_csv(path, index=False)