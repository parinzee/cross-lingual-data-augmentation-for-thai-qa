import re
from bs4 import BeautifulSoup
from pythainlp.util import normalize
import os
import psutil
import signal
import time
import multiprocessing


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