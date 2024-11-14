import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import unidecode


def setup_nltk():
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)

class Tokenizer():
    def __init__(self, lemmatize=True, remove_stop_words=True, remove_punct=True, remove_accents=True, replace_digits=False, replace_dates=False):
        self.lemmatize = lemmatize
        self.remove_stop_words = remove_stop_words
        self.remove_punct = remove_punct
        self.remove_accents = remove_accents
        self.replace_digits = replace_digits
        self.replace_dates = replace_dates

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))

        self.date_regex = re.compile(r"(\d+/\d+/\d+)")
        self.digit_regex = re.compile(r"(\d+)")

        setup_nltk()

    def tokenize(self, text):
        words = word_tokenize(text)
        result = []

        for word in words:
            if self.remove_punct and word in string.punctuation:
                continue

            if self.remove_stop_words and word in self.stop_words:
                continue

            is_date = self.date_regex.match(word)
            if self.replace_dates and is_date:
                word = "<DATE>"

            is_digit = self.digit_regex.match(word) and not is_date
            if self.replace_digits and is_digit:
                word = f"<NUMBER_{len(word)}>"

            if self.remove_accents:
                word = unidecode.unidecode(word)

            result.append(self.lemmatizer.lemmatize(word) if self.lemmatize else word)

        return result
