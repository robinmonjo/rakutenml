from bs4 import BeautifulSoup

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def preprocess_text(text):
    x = clean_html(text)
    x = x.lower()
    return x.strip()
