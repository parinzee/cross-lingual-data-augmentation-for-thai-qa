# 03 Augment LLM
# Input: `data/02_backtranslate_english.csv`
# Output: `data/03_augment_llm_gec.csv` `data/03_augment_llm_paraphrase.csv`

import os
import threading

import ai21
from ai21.errors import ServerError

import pandas as pd
import tqdm.auto as tqdm
from time import sleep

# Load api key
api_key = os.environ['AI21_API_KEY']
ai21.api_key = api_key
df = pd.read_csv('data/02_backtranslate_english.csv')

# Grammar Error Correction
def grammar_error_correction():
    # Check if output exists
    if os.path.exists('data/03_augment_llm_gec.csv'):
        data = pd.read_csv('data/03_augment_llm_gec.csv').to_dict('records')
        completed_ids = set([x["id"] for x in data])
    
    else:
        data = []
        completed_ids = set()
    
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Grammar Error Correction"):
        if row['id'] in completed_ids:
            continue
        
        # Grammar correction
        try:
            resp = ai21.GEC.execute(text=row["en_aug"])
            corrections = resp["corrections"]

            # Apply corrections
            corrected_text = row["en_aug"]
            for curr_correction in reversed(corrections):
                corrected_text = corrected_text[:curr_correction["startIndex"]] + curr_correction['suggestion'] + corrected_text[curr_correction["endIndex"]:]
        except ServerError as e:
            print(e)
            print(f"Error on id: {row['id']}")
            print(f"Error on text: {row['en_aug']}")
            print("Apending original text instead")
            corrected_text = row["en_aug"]
        
        # Append to data
        data.append({
            "id": row['id'],
            "en_llm_grammar_aug": corrected_text
        })
        
        pd.DataFrame(data).to_csv('data/03_augment_llm_gec.csv', index=False)
        sleep(2.3)

# Paraphrase
def paraphrase():
    # Check if output exists
    if os.path.exists('data/03_augment_llm_paraphrase.csv'):
        data = pd.read_csv('data/03_augment_llm_paraphrase.csv').to_dict('records')
        completed_ids = set([x["id"] for x in data])
    else:
        data = []
        completed_ids = set()
    
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Paraphrase"):
        if row['id'] in completed_ids:
            continue
        
        # Paraphrase
        resp = ai21.Paraphrase.execute(text=row["en_aug"])
        paraphrases = resp["suggestions"]
        for paraphrase in paraphrases:
            # Append to data
            data.append({
                "id": row['id'],
                "en_llm_paraphrase_aug": paraphrase["text"]
            })
        
        pd.DataFrame(data).to_csv('data/03_augment_llm_paraphrase.csv', index=False)
        sleep(1.5)
    

def run_grammar_error_correction():
    try:
        grammar_error_correction()
    except:
        # Rerun once more
        grammar_error_correction()

def run_paraphrase():
    try:
        paraphrase()
    except:
        # Rerun once more
        paraphrase()

if __name__ == "__main__":
    thread_1 = threading.Thread(target=run_grammar_error_correction) 
    thread_2 = threading.Thread(target=run_paraphrase)
    
    # Start both threads
    thread_1.start()
    thread_2.start()

    # Wait for both threads to finish
    thread_1.join()
    thread_2.join()