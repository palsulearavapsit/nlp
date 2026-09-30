"""
PIPELINE STAGE 02 - TEXT PREPROCESSING
======================================

WHAT THIS STAGE DOES
    Demonstrates the standard NLP cleaning chain on the first 10 reviews and
    prints each review BEFORE and AFTER preprocessing, so you can see what every
    step removes.

    EXPLORATORY STAGE: it prints to the console and writes no file. The later
    stages each apply their own light cleaning (a regex clean_text() function,
    or the stopword option of scikit-learn's vectorizers) rather than calling
    this script.

INPUT   amazon_reviews_100k.csv   (from stage 01)
OUTPUT  console only

NLP CONCEPTS USED
    Text cleaning   - removing noise that carries no meaning for analysis:
                      HTML tags, URLs, digits and punctuation.
    Lowercasing     - "Great" and "great" become the same token.
    Tokenization    - splitting text into individual tokens (words). Done here
                      by spaCy's tokenizer.
    Stopword removal- dropping very common function words ("the", "is", "and")
                      that appear in every review and say nothing about it.
    Lemmatization   - reducing each word to its dictionary form (lemma):
                      "running" -> "run", "batteries" -> "battery",
                      "was" -> "be". Unlike stemming (which chops suffixes, e.g.
                      "studies" -> "studi"), lemmatization always returns a real
                      word because it uses the word's part of speech.

LIBRARIES
    spaCy (model "en_core_web_sm") for tokenization, stopword flags and lemmas.
    NLTK's English stopword list is loaded but spaCy's own is_stop flag is what
    the loop below actually uses.

RUN (from the project root)
    python -m spacy download en_core_web_sm                  # once
    python -c "import nltk; nltk.download('stopwords')"     # once
    python pipeline/02_preprocessing.py
"""

import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Basic text cleaning:
    - Remove HTML
    - Remove URLs
    - Remove special characters
    - Convert to lowercase
    """

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):
    """
    Tokenization + stopword removal + lemmatization
    """

    text = clean_text(text)

    # Running spaCy's pipeline gives us a Doc: a sequence of Token objects,
    # each with attributes such as .is_stop, .is_punct and .lemma_.
    doc = nlp(text)

    tokens = []

    for token in doc:

        # Remove stopwords, spaces and punctuation
        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.is_space:
            continue

        # Lemmatization
        lemma = token.lemma_.strip()

        if lemma:
            tokens.append(lemma)

    return " ".join(tokens)


# ------------------------------------------------
# TEST ON DATASET
# ------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_100k.csv")

print("Dataset loaded!")
print("Number of reviews:", len(df))

# Combine title + review
df["text"] = df["title"].fillna("") + " " + df["review"].fillna("")

# Test preprocessing on first 10 reviews
print("\nTesting preprocessing...\n")

for i in range(10):

    original = df.loc[i, "text"]
    processed = preprocess_text(original)

    print("=" * 80)
    print("ORIGINAL:")
    print(original[:500])

    print("\nPROCESSED:")
    print(processed[:500])

print("\nPreprocessing test completed successfully!")
