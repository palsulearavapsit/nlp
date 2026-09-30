"""
PIPELINE STAGE 04 - KEYWORD EXTRACTION (TF-IDF and RAKE)
========================================================

WHAT THIS STAGE DOES
    Finds the most important terms in the corpus with two different methods
    and prints them:
        Method 1 - TF-IDF: the most important terms across ALL reviews.
        Method 2 - RAKE:   the key phrases inside one review at a time; it is
                           run on the first 5 reviews and prints each one's
                           top 10 phrases.

    EXPLORATORY STAGE: it prints to the console and writes no file. The same
    TF-IDF idea is reused for real in stage 08 (topic extraction) and in the
    app's fallback search ranker (app/semantic.py).

INPUT   amazon_reviews_100k.csv   (from stage 01)
OUTPUT  console only

NLP CONCEPT 1 - TF-IDF (Term Frequency x Inverse Document Frequency)
    A word is important to a document if it appears OFTEN in that document
    (high TF) but in FEW other documents (high IDF).
        TF(t, d)  = how many times term t occurs in document d
        IDF(t)    = log(N / df(t))   where N = number of documents and
                                     df(t) = documents that contain t
        TF-IDF    = TF x IDF
    (scikit-learn uses a smoothed IDF, ln((1 + N) / (1 + df)) + 1, and then
    scales each document's vector to length 1.)
    A word like "the" is in almost every review, so its IDF is near the
    minimum; "blu-ray" is rare, so its IDF is high. Summing each term's TF-IDF
    over all reviews ranks the terms that matter most corpus-wide.
    ngram_range=(1, 2) scores single words AND two-word phrases.

NLP CONCEPT 2 - RAKE (Rapid Automatic Keyword Extraction)
    An unsupervised method that works on a single document:
        1. Split the text into candidate phrases at stopwords and punctuation
           ("the battery life is terrible" -> "battery life", "terrible").
        2. Score each word as degree(word) / frequency(word), where degree
           counts how many words it co-occurs with inside candidate phrases.
        3. A phrase's score is the sum of its word scores, so longer, specific
           phrases rank higher.

TF-IDF vs RAKE
    TF-IDF needs a whole collection (for the IDF part) and scores terms.
    RAKE needs only one document and scores multi-word phrases.

RUN (from the project root)
    python pipeline/04_keyword_extraction.py
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import nltk

# RAKE uses NLTK's sentence tokenizer (the "punkt" model; newer NLTK versions
# call it "punkt_tab") and NLTK's English stopword list to split phrases.
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

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

df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# METHOD 1: TF-IDF
# =====================================================

print("\nRunning TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

# Shape: (100000 reviews, 10000 terms). Each cell = TF-IDF weight.
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

feature_names = vectorizer.get_feature_names_out()

print("\nTOP GLOBAL TF-IDF TERMS:")

# Corpus-level importance of each term = sum of its weights over all reviews.
scores = tfidf_matrix.sum(axis=0).A1

# argsort() sorts ascending, so take the last 30 and reverse them.
top_indices = scores.argsort()[-30:][::-1]

for i in top_indices:
    print(f"{feature_names[i]:30} {scores[i]:.2f}")


# =====================================================
# METHOD 2: RAKE
# =====================================================

print("\n\nRunning RAKE on sample reviews...")

rake = Rake()

for i in range(5):

    # RAKE runs on the ORIGINAL text: it needs the punctuation and stopwords
    # to know where one candidate phrase ends and the next begins.
    rake.extract_keywords_from_text(df.loc[i, "text"])

    phrases = rake.get_ranked_phrases_with_scores()

    print("\n" + "=" * 70)
    print("REVIEW:", df.loc[i, "text"][:300])

    print("\nTOP KEY PHRASES:")

    for score, phrase in phrases[:10]:
        print(f"{phrase:40} {score:.2f}")


print("\nKeyword extraction completed!")
