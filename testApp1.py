import streamlit as st
import spacy
import csv
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")

def preprocess(doc):
    # Remove stop words and punctuation
    tokens = [token for token in doc if not token.is_stop and not token.is_punct]
    # Lemmatize the remaining words
    lemmas = [token.lemma_ for token in tokens]
    return lemmas

def extract_features(doc):
    # Extract the sentiment of the text
    sentiment = doc.sentiment
    # Extract the frequency of certain words or phrases
    # (e.g. "fake news", "hoax", "conspiracy")
    freq = {
        "fake_news": doc.count_by(spacy.attrs.LOWER)[nlp.vocab.strings["fake news"]],
        "hoax": doc.count_by(spacy.attrs.LOWER)[nlp.vocab.strings["hoax"]],
        "conspiracy": doc.count_by(spacy.attrs.LOWER)[nlp.vocab.strings["conspiracy"]],
    }
    # Compute the readability of the text
    readability = doc._.flesch_kincaid_grade_level
     # You'll need to define this based on your chosen method

    return {
        "sentiment": sentiment,
        "frequency": freq,
        "readability": readability,
    }

with open('PolitiFact_fake_news_content.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    data = [(preprocess(nlp(row[1])), 1) for row in reader]

# Extract features from the preprocessed text
data = [(extract_features(doc), label) for doc, label in data]



