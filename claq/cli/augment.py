import os
import typer
from rich import print
from pathlib import Path
from ..crossaug import backtranslate_aug
import pandas as pd
from typing import Literal

app = typer.Typer()

@app.command()
def backtranslate(engine: Literal["google"] | Literal["opus"] = "google", source_lang: str = typer.Option("th", "--source-lang", "-s", help="Source language (default: th)"), force: bool = typer.Option(False)):
    if engine == "google":
        raise NotImplementedError("Google Translate API is not supported. Please upload into Google Sheets and translate manually instead. Otherwise, use --engine=opus")

    path = Path("data/backtranslated_dataset.csv")
    if os.path.exists(path) and not force:
        print(f"[red]{path} already exists. Use --force to overwrite.[/red]")
        raise typer.Exit(1)
    elif force:
        print(f"[red]Overwriting {path}...[/red]")
    
    # Make folder if it doesn't exist
    os.makedirs(path.parent, exist_ok=True)

    # Load dataset
    dataset = pd.read_csv("data/original_dataset.csv")
    texts = dataset["question"].tolist()

    # Augment
    print(f"[green]Augmenting {len(texts)} texts...[/green]")
    backtranslated, translated = backtranslate_aug(texts, source_lang, "en")

    # Save to csv
    augmented = dataset.copy()
    augmented[f"question_{source_lang}_aug"] = backtranslated
    augmented[f"question_en_aug"] = translated 
    augmented.to_csv(path, index=False)