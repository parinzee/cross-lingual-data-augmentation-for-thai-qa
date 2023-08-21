# 03 Augment LLM
# Input: `data/02_backtranslate_english.csv`
# Output: `data/03_augment_llm_gec.csv` `data/03_augment_llm_paraphrase.csv`

import os
import nltk
import ai21
import threading

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import pandas as pd
import tqdm.auto as tqdm
from rich import print
from functools import cache

nltk.download("punkt", quiet=True)

ai21.api_key = os.environ["AI21_API_KEY"]

# Load data
df = pd.read_csv("data/03_augment_llm_input.csv")

gec_prompt = """You are a highly skilled language model AI that returns only one line of grammatically perfect text. Your task is to evaluate the text below and correct its grammar. Even if the text is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the text is grammatically correct, do not change it. Your output should be presented WITH ONLY the corrected text IN ONE LINE and without any extra dialogue from you.

User: Phatthira Sarutpong Phokin What is the date of birth? Phatthira Sarutpong Phokin What are you playing? What is the career of Phatthira Teerathiyapong Phokin? Phatthira Sarutpong Phokin Graduated from what country? Father Cleopatra who? P?
AI: What is Phatthira Sarutpong Phokin's date of birth? What are you playing, Phatthira Sarutpong Phokin? What is Phatthira Teerathiyapong Phokin's career? Phatthira Sarutpong Phokin graduated from what country? Who is Cleopatra's father? P?

User: Emperor, Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius vibonianus gallus (206 - August 1990) Bennenus Galsen is the emperor of the Roman Empire that reigned in 1918 with the Emperor Homius and between 1918 to August. 1990, in collaboration with the son of Emperor Voluzanus?
AI: Emperor Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius Vibonianus Gallus (206 - August 1990). Bennenus Galseus is the emperor or of the Roman Empire that reigned in 1918 with Emperor Homius in 1918 and the son of Emperor Voluzanus Voluzanus in August 1990.

User: """

paraphrase_prompt = """You are a highly skilled language model AI that returns only one line of linguistically diverse paraphrased text. Your task is to perform two specific actions on a given text. First, evaluate each text and make sure it's grammatically correct. If a text is not grammatically correct, fix it. Then, ALWAYS paraphrase the text while maintaining its original meaning. Your output should be presented WITH ONLY the paraphrased text IN ONE SINGLE LINE, without any extra dialouge from you.

User: Phatthira Sarutpong Phokin What is the date of birth? Phatthira Sarutpong Phokin What are you playing? What is the career of Phatthira Teerathiyapong Phokin? Phatthira Sarutpong Phokin Graduated from what country? Who is Cleopatra's father? A? P.
AI: Phatthira Sarutpong Phokin, what is your date of birth? Phokin, What are you playing? What is Phatthira's career and from which country did Phokin graduate? Lastly, Who is the father of Cleopatra? A? P.

User: Emperor, Bennesanus Galseus or Gaisus Viyas, Bonianus Gallus, full name: Gaius vibonianus gallus (206 - August 1990) Bennenus Galsen is the emperor of the Roman Empire that reigned in 1918 with the Emperor Homius and between 1918 to August. 1990, in collaboration with the son of Emperor Voluzanus?
AI: Emperor Bennesanus Galseus, also known as Gaisus Viyas and Bonianus Gallus, had a full name of Gaius Vibonianus Gallus and reigned from 206 to August 1990. He was the Roman Emperor who collaborated with Emperor Homius in 1918 and with the latter's son Voluzanus between 1918 and August 1990.

User: """

def get_gec_prompt(text):
    return gec_prompt + text + "\nAI:"

def get_paraphrase_prompt(text):
    return paraphrase_prompt + text + "\nAI:"


@cache
@retry(wait=wait_random_exponential(min=5, max=120), stop=stop_after_attempt(3))
def gec_with_backoff(text):
    response =  ai21.Completion.execute(
        model="j2-mid",
        prompt=get_gec_prompt(text),
        numResults=1,
        maxTokens=8191,
        temperature=0.7,
        topKReturn=0,
        topP=1,
        countPenalty={
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        frequencyPenalty={
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        presencePenalty={
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        stopSequences=[]
    )
    return response["completions"][0]["data"]["text"]

@cache
@retry(wait=wait_random_exponential(min=5, max=120), stop=stop_after_attempt(3))
def paraphrase_with_backoff(text):
    response = ai21.Completion.execute(
        model="j2-mid",
        prompt=get_paraphrase_prompt(text),
        numResults=1,
        maxTokens=8191,
        temperature=0.7,
        topKReturn=0,
        topP=1,
        countPenalty={
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        frequencyPenalty={
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        presencePenalty={
            "scale": 0,
            "applyToNumbers": False,
            "applyToPunctuations": False,
            "applyToStopwords": False,
            "applyToWhitespaces": False,
            "applyToEmojis": False
        },
        stopSequences=[]
    )
    return response["completions"][0]["data"]["text"]


def batch_text(text, threshold=2500):
    # Join list of sentences into list of strings that are less than threshold characters long
    sentences = nltk.sent_tokenize(text)
    joined = []
    curr = ""
    for i, sentence in enumerate(sentences):
        if len(curr) + len(sentence) > threshold:
            joined.append(curr.strip())
            curr = ""
        curr += sentence + " "
        if i == len(sentences) - 1:
            joined.append(curr.strip())
    
    # Sanity check
    assert len(" ".join(joined)) == len(" ".join(sentences)) # no characters lost
    assert all([len(x) <= threshold for x in joined]) # no strings too long
    return joined

# Grammar Error Correction
def grammar_error_correction():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_gec.csv"):
        data = pd.read_csv("data/03_augment_llm_gec.csv").to_dict("records")
        completed_ids = set([x["context"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 15
    for i in tqdm.tqdm(range(0, len(df)), desc="Correcting grammar"):
        # Get the next 25 rows
        batch = df.iloc[i : i + 1].copy()
        batch = batch[~batch["context"].isin(completed_ids)]

        if len(batch) == 0:
            continue
            
        to_correct = batch.iloc[0]["en_aug"]
        final = ""
        for text in batch_text(to_correct):
            attempt = 0
            while True: 
                try:
                    raw_corrected = gec_with_backoff(text)
                    corrected = raw_corrected.split("\n")
                    corrected = [x.strip() for x in corrected if x.strip() != ""]
                    assert len(corrected) == len(batch)
                    break
                except AssertionError:
                    print(f"Model Returned:\n{raw_corrected}")
                    print(f"Error:\n{text}")
                    print(f"Attempt {attempt+1} of 6")
                    if attempt == 6:
                        raise
                    attempt += 1
            final += corrected[0] + " "
        
        data.append(
            {
                "context": batch.iloc[0]["context"],
                "en_llm_gec_aug": final,
            }
        )

        # Save data
        pd.DataFrame(data).to_csv("data/03_augment_llm_gec.csv", index=False)


# Paraphrase
def paraphrase():
    # Check if output exists
    if os.path.exists("data/03_augment_llm_paraphrase.csv"):
        data = pd.read_csv("data/03_augment_llm_paraphrase.csv").to_dict("records")
        completed_ids = set([x["context"] for x in data])
    else:
        data = []
        completed_ids = set()

    # Loop through each row in the dataframe using batches of 15 
    for i in tqdm.tqdm(range(0, len(df)), desc="Paraphrasing questions"):
        batch = df.iloc[i : i + 1]
        batch = batch[~batch["context"].isin(completed_ids)]

        if len(batch) == 0:
            continue
            
        to_paraphrase = batch.iloc[0]["en_aug"]
        final = ""
        for text in batch_text(to_paraphrase):
            attempt = 0
            while True: 
                try:
                    raw_paraphrase = paraphrase_with_backoff(text)
                    paraphrased = raw_paraphrase.split("\n")
                    paraphrased = [x.strip() for x in paraphrased if x.strip() != ""]
                    assert len(paraphrased) == len(batch)
                    break
                except AssertionError:
                    print(f"Model Returned:\n{raw_paraphrase}")
                    print(f"Error:\n{text}")
                    print(f"Attempt {attempt+1} of 6")
                    if attempt == 6:
                        raise
                    attempt += 1
            final += paraphrased[0] + " "

        # Add to data
        data.append(
            {"context": batch.iloc[0]["context"], "en_llm_paraphrase_aug": final}
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

    # Merge the data from both threads
    gec = pd.read_csv("data/03_augment_llm_gec.csv")
    paraphrased = pd.read_csv("data/03_augment_llm_paraphrase.csv")
    merge = pd.merge(gec, paraphrased, on="context")
    final = pd.merge(df, merge, on="context")
    
    # Sanity check that rows were not lost
    assert len(gec) == len(paraphrased)
    assert len(gec) == len(merge)
    assert len(gec) == len(final)
    assert len(final) == len(df)

    # Save the data
    final.to_csv("data/03_augment_llm.csv", index=False)
