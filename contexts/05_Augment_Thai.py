# This script may crash a couple times. Just re-run it and it should pick up where it left off.
# The crash results from memory leaks somewhere in the augmentation libraries. We're planning to fix this in the future.

import os
import gc

import pandas as pd

from tqdm import tqdm
from pythainlp.augment import word2vec, lm
from pythainlp.tokenize import word_tokenize, word_detokenize
from nltk.tokenize import sent_tokenize
from random import choice

import nlpaug.augmenter.word as naw

from utils import monitor_memory

dataset = pd.read_csv('data/05_augment_thai_input.csv')
dataset

def augment(dataset, aug_fnc, col_name, clean_every=25):
    fname_suffix = col_name.replace('th_', '')

    # Check if data already exists
    if os.path.exists(f"data/05_augment_thai_{fname_suffix}.csv"):
        data = pd.read_csv(f"data/05_augment_thai_{fname_suffix}.csv").to_dict("records")
        completed_ids = set([x["id"] for x in data])
    else:
        data = []
        completed_ids = set()

    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), miniters=0):
        if not row['id'] in completed_ids:
            if row['context'][0] not in ['า']:
                chunks = sent_tokenize(row[""])
                augmented = aug_fnc(row['question'])

                # Randomly select one of the augmented sentences
                sent = choice(augmented)
                sent = ''.join(sent)
                data.append({
                    'id': row['id'],
                    col_name: sent,
                })
                del sent
                del augmented

                # Try to prevent memory leaks as much as possible
                if idx % clean_every == 0:
                    gc.collect()
                
                # Save data
                pd.DataFrame(data).to_csv(f"data/05_augment_thai_{fname_suffix}.csv", index=False)
            else:
                # Incomplete questions causes the augmentation libraries to crash
                data.append({
                    'id': row['id'],
                    col_name: row['question'],
                })
                pd.DataFrame(data).to_csv(f"data/05_augment_thai_{fname_suffix}.csv", index=False)

    return pd.DataFrame(data)

if __name__ == "__main__":
    mm_process = monitor_memory(0.75) # Quit program if memory usage exceeds 75% of total RAM

    # Wordnet
    print('Performing WordNet augmentation...')
    wordnet = naw.SynonymAug(aug_src='wordnet', lang='tha', tokenizer=word_tokenize, reverse_tokenizer=word_detokenize)
    wordnet_aug = augment(dataset, lambda x: wordnet.augment(x, n=1), 'th_wordnet_aug', clean_every=100)

    # Thai2Fit
    print('Performing Thai2Fit augmentation...')
    thai2fit = word2vec.Thai2fitAug()
    thai2fit_aug = augment(dataset, lambda x: thai2fit.augment(x, n_sent=1), 'th_thai2fit_aug')

    # LTW2VecAug
    print('Performing LTW2Vec augmentation...')
    ltw2v = word2vec.LTW2VAug()
    ltw2v_aug = augment(dataset, lambda x: ltw2v.augment(x, n_sent=1), 'th_ltw2v_aug')

    # Fast Text Aug
    print('Performing FastText augmentation...')
    if not os.path.exists('cc.th.300.vec'):
        # Download FastText model
        os.system('curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.th.300.vec.gz')
        os.system('gunzip cc.th.300.vec.gz')
    
    fasttext = naw.WordEmbsAug(model_type='fasttext', tokenizer=word_tokenize, reverse_tokenizer=word_detokenize,
                      model_path="cc.th.300.vec")
    fasttext_aug = augment(dataset, lambda x: fasttext.augment(x, n=1), 'th_fasttext_aug', clean_every=100)

    # Merge all augmented data
    augmented_data = dataset.copy()
    augmented_data = pd.merge(augmented_data, wordnet_aug, on='id')
    augmented_data = pd.merge(augmented_data, thai2fit_aug, on='id')
    augmented_data = pd.merge(augmented_data, ltw2v_aug, on='id')
    augmented_data = pd.merge(augmented_data, fasttext_aug,  on='id')
    augmented_data.to_parquet('data/05_augment_thai.parquet', index=False)
    
    # Clean up
    mm_process.terminate()
    exit()