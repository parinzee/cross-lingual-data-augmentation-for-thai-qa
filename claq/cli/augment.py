import os
import typer
from rich import print
from pathlib import Path
from ..crossaug import backtranslate_aug
import pandas as pd
from typing import Literal, Union
from enum import Enum

app = typer.Typer()

class BacktranslationEngine(str, Enum):
    google = "google"
    opus = "opus"

@app.command()
def backtranslate(engine: BacktranslationEngine = "google", path: str = "data/backtranslate.parquet", original_path: str = "data/original_dataset.parquet", source_lang: str = typer.Option("th", "--source-lang", "-s", help="Source language (default: th)"), force: bool = typer.Option(False)):
    if engine == "google":
        if not os.path.exists("data/temp/translated_questions.csv") or not os.path.exists("data/temp/translated_context.csv"):
            data = pd.read_parquet(original_path)
            export_q = data[["id", "question"]]
            export_c = data[["id", "context"]]

            # Make folder if doesn't exist
            Path("data/temp").mkdir(parents=True, exist_ok=True)

            export_q.to_csv("data/temp/to_translate_question.csv", index=False)
            print("Exported questions to data/temp/to_translate_question.csv")
            export_c.to_csv("data/temp/to_translate_context.csv", index=False)
            print("Exported context to data/temp/to_translate_context.csv\n")

            print("[bold green]Translate with google sheets into columns: [/bold green]")
            print(" - question: question_en_aug")
            print(" - context: context_en_aug\n")
            print("Place translated data into [bold blue]data/temp/translated_question.csv [/bold blue] and [bold blue]data/temp/translated_context.csv[/bold blue]")
            print("Then, rerun this command")
        else:
            export_q = pd.read_csv("data/temp/translated_question.csv")
            export_c = pd.read_csv("data/temp/translated_context.csv")

            # Merge with original dataset
            data = pd.read_parquet(original_path)
            data = data.merge(export_q, on="id")
            data = data.merge(export_c, on="id")
            
            # Save to parquet
            data.to_parquet(path, index=False)

    else:
        path = Path(path)
        if os.path.exists(path) and not force:
            print(f"[red]{path} already exists. Use --force to overwrite.[/red]")
            raise typer.Exit(1)
        elif force:
            print(f"[red]Overwriting {path}...[/red]")
        
        # Make folder if it doesn't exist
        os.makedirs(path.parent, exist_ok=True)

        # Load dataset
        dataset = pd.read_parquet("data/original_dataset.parquet")
        texts = dataset["question"].tolist()

        # Augment
        print(f"[green]Augmenting {len(texts)} texts...[/green]")
        backtranslated, translated = backtranslate_aug(texts, source_lang, "en")

        # Save to csv
        augmented = dataset.copy()
        augmented[f"question_{source_lang}_aug"] = backtranslated
        augmented[f"question_en_aug"] = translated 
        augmented.to_csv(path, index=False)