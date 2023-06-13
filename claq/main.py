import typer
from .cli import data

app = typer.Typer()
app.add_typer(data.app, name="data")

if __name__ == "__main__":
    app()