"""Base class for all augmentations."""
import gc
from typing import List

from tqdm.autonotebook import tqdm

class BaseAugment():
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.clean_every = 1000
        self.cache = {}
    
    def _augment(self, text: str):
        raise NotImplementedError

    def augment(self, text: str):
        if text not in self.cache:
            self.cache[text] = self._augment(text)
        return self.cache[text]
    
    def augment_batch(self, texts: List[str]):
        augmented = []
        for i, t in tqdm(enumerate(texts), desc=f"Performing {self.__class__.__name__} augmentation:", total=len(texts)):
            augmented.append(self.augment(t))
            if i % self.clean_every == 0:
                gc.collect()
        return augmented

    def __call__(self, text: str):
        return self.augment(text)
    
    def __repr__(self):
        return self.__class__.__name__ + f"({self._args}, {self._kwargs})"
    
    def __str__(self):
        return self.__repr__()
