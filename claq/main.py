import os
import logging

import typer
from rich import print

from .cli import data
from .cli import augment


state = {
    # Check for each dataset stages
    "original_dataset": False,
    "backtranslated_dataset": False,
    "monoaug_dataset": False,
    "llmaug_dataset": False,
    "crossaug_dataset": False,
    "final_dataset": False,
}

def callback(ctx: typer.Context, debug: bool = typer.Option(False)):
    # Check state
    for key in state:
        # Files may end in parquet or csv
        state[key] = os.path.exists(os.path.join("data", f"{key}.csv")) or os.path.exists(os.path.join("data", f"{key}.parquet"))

    if ctx.invoked_subcommand is None:
        print("[bold blue]Welcome to CLAQ![/bold blue]")
        # Print each dataset stages
        print("[yellow][bold]Dataset stages:[/yellow][/bold]")
        for stage, status in state.items():
            print(f"  - [bold]{stage}[/bold]: {'[green]Done[/green]' if status else '[red]Not done[/red]'}")

        typer.echo("\nUse --help to see available commands.")
    
    # Check for debug flag
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        print("[red]Debug mode enabled.[/red]")

app = typer.Typer(callback=callback, invoke_without_command=True)
app.add_typer(data.app, name="data", callback=callback, invoke_without_command=True)
app.add_typer(augment.app, name="augment", callback=callback, invoke_without_command=True)

if __name__ == "__main__":
    app()