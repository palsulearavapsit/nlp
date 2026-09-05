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