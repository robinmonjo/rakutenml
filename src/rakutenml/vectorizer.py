import numpy as np

from .tokenizer import Tokenizer

class Vectorizer():
    def __init__(self, vector_size=None, tokenizer=None):
        self.vector_size = vector_size
        self.vocabulary = {}
        self.tokenizer = tokenizer or Tokenizer(lemmatize=False, remove_stop_words=True, remove_punct=True, replace_digits=True, replace_dates=True)

    def fit(self, documents):
        return self.fit_tokens(self.tokenize(documents))

    def fit_tokens(self, tokens):
        self.vocabulary = {}
        for ts in tokens:
            sub_ts = ts if self.vector_size == None else ts[:self.vector_size+1]
            for t in sub_ts:
                if t not in self.vocabulary:
                    self.vocabulary[t] = len(self.vocabulary) + 1 # keep 0 for padding
        self.vocabulary["<UNK>"] = len(self.vocabulary) + 1
        return self

    def tokenize(self, documents):
        return [self.tokenizer.tokenize(document) for document in documents]

    def vectorize(self, documents):
        return self.vectorize_tokens(self.tokenize(documents))

    def vectorize_tokens(self, tokens):
        vectors = [[self.vocabulary.get(t) or self.vocabulary.get("<UNK>") for t in ts] for ts in tokens]
        return self.pad_vectors(vectors)

    def pad_vectors(self, vectors):
        max_len = self.vector_size
        if max_len is None:
            max_len = max(len(v) for v in vectors)

        padded_vectors = []
        for v in vectors:
            if len(v) > max_len:
                padded_v = v[:max_len]
            else:
                padded_v = v + [0] * (max_len - len(v))

            padded_vectors.append(padded_v)

        return np.array(padded_vectors)
