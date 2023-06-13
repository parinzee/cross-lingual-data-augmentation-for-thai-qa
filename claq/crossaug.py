import logging

import torch
from torch.utils.data import Dataset
from transformers import pipeline

from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

# Check for cuda, check for mps, otherwise use cpu
if torch.cuda.is_available():
    logger.info("Setting default device to cuda:0")
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    logger.info("Setting default device to mps")
    device = torch.device("mps")
else:
    logger.info("Setting default device to cpu")
    device = torch.device("cpu")

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def translate(texts, source_lang, target_lang, device=device, batch_size=16):
    logger.info(f"Translating {len(texts)} texts from {source_lang} to {target_lang}...")
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}", device=device)

    dataset = ListDataset(texts)

    # Translate to target language
    translated = []
    for out in tqdm(translator(dataset, batch_size=batch_size), total=len(dataset), miniters=0):
        translated.append(out)

    return [t["translation_text"] for t in translated]

def backtranslate_aug(texts, source_lang, target_lang, device=device, batch_size=16):
    # Translate to target language
    logger.info(f"Backtranslating {len(texts)} texts from {source_lang} to {target_lang} and back to {source_lang}...")
    translated = translate(texts, source_lang, target_lang, device=device, batch_size=batch_size)
    backtranslated = translate(translated, target_lang, source_lang, device=device, batch_size=batch_size)
    return backtranslated, translated