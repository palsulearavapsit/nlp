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

X1 = vectorizer_1.fit_transform(df["clean_text"])

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