import numpy as np
from backend.model.vocab import Vocab

class Tokenizer:
    def __init__(self, vocab: Vocab, max_len: int):
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get_index(token) for token in tokens]
        return ids[:self.max_len]

    def pad_sequence(self, seq: list[int]) -> list[int]:
        if len(seq) < self.max_len:
            seq = seq + [self.vocab.pad_id] * (self.max_len - len(seq))
        return seq[:self.max_len]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.pad_sequence(self.encode(text)) for text in texts])