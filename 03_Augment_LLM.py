# 03 Augment LLM
# Input: `data/02_backtranslate_english.csv`
# Output: `data/03_augment_llm_gec.csv` `data/03_augment_llm_paraphrase.csv`

import os
import threading

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import pandas as pd
import tqdm.auto as tqdm
from rich import print

# Load api key
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

# Load data
df = pd.read_csv("data/02_backtranslate_english.csv")

# AI Constants
# These original messages are generated from the same model, however we include them inside the chat to make sure the model returns in the correct format.
model = "gpt-3.5-turbo-0301"
example_grammar_error_correction_messages = [
    {
        "role": "system",
        "content": "You are a highly skilled language model AI. Your task is to evaluate each sentence from a given list and correct its grammar. Even if a sentence is not clear or intelligible, ALWAYS make a grammatical correction, even if you have to make assumptions about the intended meaning. If the sentence is grammatically correct, do not change it. Your output should be presented with ONLY each corrected sentence written on a new line.",
    },
    {
        "role": "user",
        "content": "Phatthira Sarutpong Phokin What is the date of birth?\nPhatthira Sarutpong Phokin What are you playing?\nWhat is the career of Phatthira Teerathiyapong Phokin?\nPhatthira Sarutpong Phokin Graduated from what country?\nWho is Cleopatra's father?",
    },
    {
        "role": "assistant",
        "content": "What is Phatthira Sarutpong Phokin's date of birth?\nWhat are you playing, Phatthira Sarutpong Phokin?\nWhat is Phatthira Teerathiyapong Phokin's career?\nPhatthira Sarutpong Phokin graduated from what country?\nWho is Cleopatra's father?",
    },
]

example_paraphrase_messages = [
    {
        "role": "system",
        "content": "You are a highly skilled language model AI. Your task is to perform two specific actions on a given list of sentences. First, evaluate each sentence and make sure it's grammatically correct. If a sentence is not grammatically correct, fix it. Then, ALWAYS paraphrase each sentence while maintaining its original meaning.  Your output should be presented with ONLY each paraphrased sentence written on a new line.",
    },
    {
        "role": "user",
        "content": "Phatthira Sarutpong Phokin What is the date of birth?\nPhatthira Sarutpong Phokin What are you playing?\nWhat is the career of Phatthira Teerathiyapong Phokin?\nPhatthira Sarutpong Phokin Graduated from what country?\nWho is Cleopatra's father?",
    },
    {
        "role": "assistant",
        "content": "Phatthira Sarutpong Phokin, what is your date of birth?\nWhat are you playing, Phatthira Sarutpong Phokin?\nWhat is Phatthira Teerathiyapong Phokin's career?\nFrom which country did Phatthira Sarutpong Phokin graduate?\nWho is the father of Cleopatra?",
    },
]


# Add exponential backoff to completion
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


# Grammar Error Correction
def grammar_error_correction():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_gec.csv"):
        data = pd.read_csv("data/03_augment_llm_gec.csv").to_dict("records")
        completed_ids = set([x["id"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 25
    for i in tqdm.tqdm(range(0, len(df), 15), desc="Correcting grammar"):
        # Get the next 25 rows
        batch = df.iloc[i : i + 15]
        batch = batch[~batch["id"].isin(completed_ids)]
        if len(batch) == 0:
            continue

        to_correct = "\n".join(batch["en_aug"].tolist())
        attempt = 0
        while attempt < 3:
            try:
                response = completion_with_backoff(
                    model=model,
                    messages=[
                        *example_grammar_error_correction_messages,
                        {"role": "user", "content": to_correct},
                    ],
                )
                raw_corrected = response["choices"][0]["message"]["content"]
                corrected = raw_corrected.split("\n")
                assert len(corrected) == len(batch)
                break
            except AssertionError:
                print(f"Error: {raw_corrected}")
                print(f"Error: {batch}")
                print(f"Attempt {attempt+1} of 3")
                if attempt == 3:
                    raise

        # Add to data
        for idx, corrected_sentence in enumerate(corrected):
            data.append(
                {
                    "id": batch.iloc[idx]["id"],
                    "en_llm_gec_aug": corrected_sentence,
                }
            )

        # Save data
        pd.DataFrame(data).to_csv("data/03_augment_llm_gec.csv", index=False)


# Paraphrase
def paraphrase():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_paraphrase.csv"):
        data = pd.read_csv("data/03_augment_llm_paraphrase.csv").to_dict("records")
        completed_ids = set([x["id"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 25
    for i in tqdm.tqdm(range(0, len(df), 25), desc="Paraphrasing questions"):
        # Get the next 25 rows
        batch = df.iloc[i : i + 25]
        batch = batch[~batch["id"].isin(completed_ids)]
        if len(batch) == 0:
            continue

        to_paraphrase = "\n".join(batch["en_aug"].tolist())
        attempt = 0
        while attempt < 3:
            try:
                response = completion_with_backoff(
                    model=model,
                    messages=[
                        *example_paraphrase_messages,
                        {"role": "user", "content": to_paraphrase},
                    ],
                )
                raw_paraphrases = response["choices"][0]["message"]["content"]
                paraphrases = raw_paraphrases.split("\n")
                assert len(paraphrases) == len(batch)
                break
            except AssertionError:
                print(f"Error: {raw_paraphrases}")
                print(f"Error: {batch}")
                print(f"Attempt {attempt+1} of 3")
                if attempt == 3:
                    raise

        # Add to data
        for idx, paraphrased in enumerate(paraphrases):
            data.append(
                {"id": batch.iloc[idx]["id"], "en_llm_paraphrase_aug": paraphrased}
            )
        pd.DataFrame(data).to_csv("data/03_augment_llm_paraphrase.csv", index=False)


if __name__ == "__main__":
    thread_1 = threading.Thread(target=grammar_error_correction)
    thread_2 = threading.Thread(target=paraphrase)

    # Start both threads
    thread_1.start()
    thread_2.start()

    # Wait for both threads to finish
    thread_1.join()
    thread_2.join()
