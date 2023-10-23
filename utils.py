import random
import torch
import numpy as np
import hashlib
from collections import defaultdict
import re
from bs4 import BeautifulSoup
from pythainlp.util import normalize
import os
import psutil
import signal
import time
import multiprocessing
from functools import lru_cache
from tqdm.auto import tqdm
import numpy as np
import gc



def seed_everything(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_row_to_simple_transformers_format(row, question_col="question"):
    # Initialize an empty list to store converted answers
    converted_answers = []

    # Iterate over each answer and its corresponding start index
    for i in range(len(row['answers']['text'])):
        # Create a dictionary for the current answer
        answer_dict = {
            'text': row['answers']['text'][i],
            'answer_start': row['answers']['answer_start'][i],
        }

        # Add the current answer to the list of converted answers
        converted_answers.append(answer_dict)

    # Simpletransformers requires that ids be unique
    # If we augment the questions, the current id scheme would collide
    # Thus we rehash the id with the current question instead
    id = hashlib.sha256((row["context"] + row[question_col]).encode("utf-8")).hexdigest()

    # Create a dictionary for the question and answers
    qas_dict = {
        'id': id,
        "is_impossible": False,
        'question': row[question_col],
        'answers': converted_answers
    }

    # Wrap the 'context', 'question', and 'answers' into a 'qas' list
    # Return the converted example
    return {'context': row['context'], 'qas': [qas_dict]}

def merge_qas(data):
    merged_data = defaultdict(list)

    # Loop over each dictionary in the dataset
    for item in data:
        # Add each 'qas' to the list of 'qas' for the same 'context'
        merged_data[item['context']].extend(item['qas'])

    # Convert the merged_data back into the original format: a list of dictionaries
    merged_data = [{'context': context, 'qas': qas} for context, qas in merged_data.items()]

    return merged_data

def clean_text(text, is_question=False):
    # Remove html tags
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()

    # Remove semicolons
    text = re.sub(r';', '', text)

    # Remove empty parenthesis and parenthesis with only whitespace inside
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\(;\s*"(\w+)"\)', r'("\1")', text)

    # Remove reference citations for example [2]:7 or [9]:5 (present in tydiqa)
    text = re.sub(r'\[\d+\]:\d+', '', text)
    text = re.sub(r'\[\d+\]', '', text)

    # Remove more than one whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip text inside of parenthesis
    text = re.sub(r'\(\s*([^)]*)\)', r'(\1)', text)

    # Remove em dashes
    text = re.sub(u"\u2014", "", text)

    # Remove whitespace
    text = text.strip()

    # If question, remove question mark and strip since some questions have whitespace between the question mark and the end of the question
    if is_question:
        text = re.sub(r'\?', '', text)
        text = text.strip()
        text = text + "?"
    
    # Pythainlp normalize
    text = normalize(text)

    return text


def _monitor_memory(main_pid, threshold):
    while True:
        main_process = psutil.Process(main_pid)
        memory_usage = main_process.memory_info().rss
        total_memory = psutil.virtual_memory().total

        if memory_usage > total_memory * threshold:
            os.kill(main_pid, signal.SIGTERM)
            exit(-1)

        time.sleep(0.01)

def monitor_memory(threshold):
    main_pid = os.getpid()
    memory_monitor = multiprocessing.Process(target=_monitor_memory, args=(main_pid, threshold))
    memory_monitor.start()
    return memory_monitor

# define global cache dictionaries
cache_whole_texts = {}
cache_batches = {}

def encode_in_batch(model, texts, progress=True):
    # cache for whole texts
    texts_tuple = tuple(texts) # since lists can't be dict keys
    if texts_tuple in cache_whole_texts:
        return cache_whole_texts[texts_tuple]

    batch_size = 256
    all_embeddings = []

    if progress:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            # cache for batches
            batch_tuple = tuple(batch_texts)
            if batch_tuple in cache_batches:
                embeddings = cache_batches[batch_tuple]
            else:
                embeddings = cache_individual_texts(model, *batch_texts)
                cache_batches[batch_tuple] = np.asarray(embeddings)
            all_embeddings.append(embeddings)
    else:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            # cache for batches
            batch_tuple = tuple(batch_texts)
            if batch_tuple in cache_batches:
                embeddings = cache_batches[batch_tuple]
            else:
                embeddings = cache_individual_texts(model, *batch_texts)
                cache_batches[batch_tuple] = np.asarray(embeddings)
            all_embeddings.append(embeddings)
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # save to whole texts cache
    cache_whole_texts[texts_tuple] = all_embeddings

    gc.collect()

    return all_embeddings

@lru_cache(maxsize=None) # cache for individual texts
def cache_individual_texts(model, *batch_texts):
    batch_texts = list(batch_texts)
    for item in batch_texts:
        try:
            assert type(item) == str
        except:
            print(item)
            raise
    embeddings = model(list(batch_texts))
    return embeddings
    
def list_to_ordered_set(input_list):
    seen = set()
    output_list = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            output_list.append(item)
    
    del seen
            
    return output_list