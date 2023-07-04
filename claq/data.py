import re
import hashlib
import logging
from copy import deepcopy

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm
from pythainlp.util import normalize
from datasets import load_dataset as hf_load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)

def clean_text(text, is_question=False):
    # Remove html tags
    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text()

    # Remove semicolons
    text = re.sub(r";", "", text)

    # Remove empty parenthesis and parenthesis with only whitespace inside
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r'\(;\s*"(\w+)"\)', r'("\1")', text)

    # Remove reference citations for example [2]:7 or [9]:5 (present in tydiqa)
    text = re.sub(r"\[\d+\]:\d+", "", text)
    text = re.sub(r"\[\d+\]", "", text)

    # Remove more than one whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip text inside of parenthesis
    text = re.sub(r"\(\s*([^)]*)\)", r"(\1)", text)

    # Remove em dashes
    text = re.sub("\u2014", "", text)

    # Remove whitespace
    text = text.strip()

    # If question, remove question mark and strip since some questions have whitespace between the question mark and the end of the question
    if is_question:
        text = re.sub(r"\?", "", text)
        text = text.strip()
        text = text + "?"

    # Pythainlp normalize
    text = normalize(text)

    return text


def merge_dataset_splits(dataset):
    splits = list(dataset.keys())
    if len(splits) == 1:
        return dataset[splits[0]]
    else:
        return concatenate_datasets([dataset[split] for split in splits])


# Datasets require more processing to make them all in the correct format
def get_offset_begin_position(cleaned_context, answers_text, answer_begin_positions):
    new_answer_begin_positions = []

    try:
        for answer_text, answer_begin_position in zip(
            answers_text, answer_begin_positions
        ):
            # Find all all instances of the answer in cleaned_context
            possible_answer_begin_positions = [
                i
                for i in range(len(cleaned_context))
                if cleaned_context.startswith(answer_text, i)
            ]

            if len(possible_answer_begin_positions) == 1:
                new_answer_begin_positions.append(possible_answer_begin_positions[0])
            elif len(possible_answer_begin_positions) > 1:
                # Find the closest answer to the original answer
                closest_answer_begin_position = min(
                    possible_answer_begin_positions,
                    key=lambda x: abs(x - answer_begin_position),
                )
                new_answer_begin_positions.append(closest_answer_begin_position)
            else:
                raise Exception("No answer found in context")
    except:
        logger.critical("Error with answer: %s", str(answer_text))
        logger.critical("Error with context: %s", str(cleaned_context))
        logger.critical(
            "Error with answer begin position: %s", str(answer_begin_position)
        )
        raise

    return new_answer_begin_positions


def process_row(row, answer_text_key, answer_start_key):
    # Normalize the text
    new_row = deepcopy(row)
    new_row["context"] = clean_text(row["context"])
    new_row["question"] = clean_text(row["question"], is_question=True)

    new_row["answers"] = {}
    new_row["answers"]["text"] = [
        clean_text(x) for x in row["answers"][answer_text_key]
    ]

    # Reindex the dataset
    new_row["answers"]["answer_start"] = get_offset_begin_position(
        new_row["context"], new_row["answers"]["text"], row["answers"][answer_start_key]
    )
    new_row["answers"]["answer_end"] = [
        x + len(y)
        for x, y in zip(new_row["answers"]["answer_start"], new_row["answers"]["text"])
    ]

    return new_row


def sanity_check(datasets):
    logger.info("Initiating sanity check of processed datasets")

    # Match keys
    logger.info("Matching keys")
    for dataset in tqdm(datasets):
        assert list(dataset.columns) == list(datasets[0].columns)
        assert dataset["answers"][0].keys() == datasets[0]["answers"][0].keys()
    logger.info("Matched keys")

    # Check theortical answers vs index
    logger.info("Matching theoretical answers vs indexed answers")
    for dataset in datasets:
        for _, row in tqdm(list(dataset.iterrows())):
            for text, begin, end in zip(
                row["answers"]["text"],
                row["answers"]["answer_start"],
                row["answers"]["answer_end"],
            ):
                assert (
                    text == row["context"][begin:end]
                ), f"Theoretical Answer: {text} | Indexed: {row['context'][begin:end]} | Context: {row['context']}"
    logger.info("Matched theoretical answers vs indexed answers")

    # Assert no empty columns
    logger.info("Checking for empty columns")
    for dataset in tqdm(datasets):
        assert not dataset.isnull().values.any()
    logger.info("Checked for empty columns")

    logger.info("Sanity check passed")


def generate_id(row):
    return hashlib.sha256(
        (row["context"] + row["question"]).encode("utf-8")
    ).hexdigest()


def load_unprocessed_qa_dataset() -> pd.DataFrame:
    logger.info("Loading datasets")
    iapp = hf_load_dataset("iapp_wiki_qa_squad")
    thaiqa = hf_load_dataset("thaiqa_squad")
    xquad = hf_load_dataset("xquad", "xquad.th")
    tydiqa = hf_load_dataset("khalidalt/tydiqa-goldp", "thai")
    logger.info("Loaded datasets")

    # Process the datasets
    logger.info("Merging dataset splits")
    iapp = merge_dataset_splits(iapp).to_pandas()
    thaiqa = merge_dataset_splits(thaiqa).to_pandas()
    xquad = merge_dataset_splits(xquad).to_pandas()
    tydiqa = merge_dataset_splits(tydiqa).to_pandas()
    logger.info("Merged dataset splits")

    # Get only required columns
    iapp = iapp[['question', 'context', 'answers']]
    thaiqa = thaiqa[['question', 'context', 'answers']]
    xquad = xquad[['question', 'context', 'answers']]
    tydiqa = tydiqa.rename(columns={'passage_text': 'context', "question_text": "question"})[['question', 'context', 'answers']]

    # Specially processing tydiqa (see notebook 01 for more info)
    logger.warning("Special processing for tydiqa dataset required, performing now")
    tydiqa = tydiqa.drop([480, 686, 953, 2057, 3177])
    tydiqa = tydiqa[
        tydiqa["answers"].apply(lambda x: not any(["}'>" in y for y in x["text"]]))
    ]
    tydiqa = tydiqa[
        tydiqa["answers"].apply(lambda x: not any([":[" in y for y in x["text"]]))
    ]
    tydiqa = tydiqa[
        tydiqa["answers"].apply(lambda x: not any(["=" in y for y in x["text"]]))
    ]
    tydiqa = tydiqa[
        tydiqa["answers"].apply(lambda x: not any(["[" in y for y in x["text"]]))
    ]
    logger.warning("Special processing for tydiqa completed")

    # Cleaning datasets
    logger.info("Cleaning datasets")
    iapp = iapp.apply(lambda x: process_row(x, "text", "answer_start"), axis=1)
    thaiqa = thaiqa.apply(
        lambda x: process_row(x, "answer", "answer_begin_position"), axis=1
    )
    xquad = xquad.apply(lambda x: process_row(x, "text", "answer_start"), axis=1)
    tydiqa = tydiqa.apply(lambda x: process_row(x, "text", "start_byte"), axis=1)
    logger.info("Cleaned datasets")

    sanity_check([iapp, thaiqa, xquad, tydiqa])

    # Merging datasets
    logger.info("Merging all datasets")
    iapp["source"] = "iapp"
    thaiqa["source"] = "thaiqa"
    xquad["source"] = "xquad"
    tydiqa["source"] = "tydiqa"
    final = pd.concat([iapp, thaiqa, xquad, tydiqa], ignore_index=True)
    logger.info("Merged all dataset")

    # Removing duplicates
    logger.info("Dropping duplicates")
    logger.info("Length before dropping duplicates: %s", str(len(final)))

    final = final.drop_duplicates(
        subset=["context", "question"], keep="first"
    ).reset_index(drop=True)

    logger.info("Length after dropping duplicates: %s", str(len(final)))
    logger.info("Dropped duplicates")

    # Generate unique ids
    logger.info("Generating unique ids for the data")
    final["id"] = final.apply(generate_id, axis=1)
    logger.info("Generated unique ids for the data")

    return final
