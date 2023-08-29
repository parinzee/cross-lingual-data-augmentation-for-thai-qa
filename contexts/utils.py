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
    