"""
PIPELINE STAGE 03 - N-GRAM ANALYSIS
===================================

WHAT THIS STAGE DOES
    Counts the most frequent single words, word pairs and word triples across
    all 100,000 reviews and prints the top 20 of each.

    EXPLORATORY STAGE: it prints to the console and writes no file. Its
    findings (for example that "waste money" and "highly recommend" are among
    the most frequent bigrams) informed the hand-built phrase lexicon used by
    the live sentiment engine in app/lexicon.py.

INPUT   amazon_reviews_100k.csv   (from stage 01)
OUTPUT  console only

NLP CONCEPT - N-GRAMS
    An n-gram is a run of n consecutive words:
        unigram (n=1): "waste"
        bigram  (n=2): "waste money"
        trigram (n=3): "complete waste money"   (stopwords like "of" removed)
    Single words lose context: "recommend" alone could be part of
    "do not recommend". Counting bigrams and trigrams captures common phrases
    whose meaning comes from the words together.

HOW IT WORKS
    1. Clean the text (lowercase, drop URLs, digits and punctuation).
    2. scikit-learn's CountVectorizer tokenizes every review, removes English
       stopwords and builds a document-term matrix: one row per review, one
       column per n-gram, each cell = how many times that n-gram occurs.
       ngram_range=(2, 2) means "bigrams only", etc.
       max_features=10000 keeps only the 10,000 most frequent n-grams.
    3. Summing each column gives the total count of that n-gram in the corpus.
    4. Sort by count and print the top 20.

RUN (from the project root)
    python pipeline/03_ngram_analysis.py
"""

import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_100k.csv")

# Combine title + review
df["text"] = (
    df["title"].fillna("") + " " +
    df["review"].fillna("")
)

# Basic cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")

df["clean_text"] = df["text"].apply(clean_text)

# ---------------------------------------------------
# UNIGRAMS
# ---------------------------------------------------

print("\nExtracting unigrams...")

vectorizer_1 = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 1),
    max_features=10000
)

# X1 is a sparse matrix of shape (100000 reviews, 10000 unigrams).
X1 = vectorizer_1.fit_transform(df["clean_text"])

# Column sums = total occurrences of each unigram across the whole corpus.
# .A1 flattens the 1 x N matrix into a plain 1-D array.
unigram_counts = X1.sum(axis=0).A1
unigram_words = vectorizer_1.get_feature_names_out()

unigrams = sorted(
    zip(unigram_words, unigram_counts),
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 20 UNIGRAMS")

for word, count in unigrams[:20]:
    print(f"{word:25} {count}")


# ---------------------------------------------------
# BIGRAMS
# ---------------------------------------------------

print("\nExtracting bigrams...")

vectorizer_2 = CountVectorizer(
    stop_words="english",
    ngram_range=(2, 2),
    max_features=10000
)

X2 = vectorizer_2.fit_transform(df["clean_text"])

bigram_counts = X2.sum(axis=0).A1
bigram_words = vectorizer_2.get_feature_names_out()

bigrams = sorted(
    zip(bigram_words, bigram_counts),
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 20 BIGRAMS")

for phrase, count in bigrams[:20]:
    print(f"{phrase:35} {count}")


# ---------------------------------------------------
# TRIGRAMS
# ---------------------------------------------------

print("\nExtracting trigrams...")

vectorizer_3 = CountVectorizer(
    stop_words="english",
    ngram_range=(3, 3),
    max_features=10000
)

X3 = vectorizer_3.fit_transform(df["clean_text"])

trigram_counts = X3.sum(axis=0).A1
trigram_words = vectorizer_3.get_feature_names_out()

trigrams = sorted(
    zip(trigram_words, trigram_counts),
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 20 TRIGRAMS")

for phrase, count in trigrams[:20]:
    print(f"{phrase:45} {count}")

print("\nN-gram analysis completed successfully!")
