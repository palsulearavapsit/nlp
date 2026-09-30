"""
PIPELINE STAGE 08 - TOPIC EXTRACTION (TF-IDF + NMF topic modelling)
===================================================================

WHAT THIS STAGE DOES
    Discovers 10 themes ("topics") that run through the 100,000 reviews without
    being told what to look for, then tags every review with its main topic.

INPUT
    amazon_reviews_sentiment.csv     (from stage 06)
OUTPUT
    amazon_reviews_topics.csv        columns: title, review, vader_sentiment,
                                              vader_score, topic (1..10)
    topic_keywords.csv               columns: topic, keywords
                                     (the 10 most important terms per topic)

    The app reads both files for the "Topics" page - see app/server.py ->
    topic_list() and topic_reviews().

NLP CONCEPT - TOPIC MODELLING
    Topic modelling is UNSUPERVISED: there are no labels. It assumes each
    document is a mixture of a few hidden topics, and each topic is a group of
    words that tend to appear together ("album, songs, band, rock").

HOW IT WORKS
    1. Clean the text (lowercase, drop URLs, digits and punctuation).
    2. TF-IDF turns the corpus into a matrix X of shape (100,000 reviews x
       10,000 terms). See stage 04 for what TF-IDF means. Settings:
           ngram_range=(1, 2)  single words and two-word phrases
           min_df=5            ignore terms in fewer than 5 reviews (typos)
           max_df=0.90         ignore terms in more than 90% of reviews
    3. NMF (Non-negative Matrix Factorization) splits X into two smaller
       matrices whose product approximates it:
                     X        ~        W        x        H
           (reviews x terms)    (reviews x 10)    (10 x terms)
       * Each row of H is a TOPIC: a weight for every term. Its highest-
         weighted terms are the topic's keywords.
       * Each row of W says how strongly a review belongs to each topic.
       "Non-negative" means every weight is >= 0, so a review is an ADDITIVE
       mix of topics. That makes the topics easy for people to read.
    4. Each review's topic = the topic with the largest weight in its row of W
       (argmax). +1 makes topics numbered 1..10 instead of 0..9.

WHY NMF (and not LDA)?
    LDA is the other classic topic model. It is a probabilistic model normally
    run on raw word counts. NMF works directly on TF-IDF weights, is fast on
    100k documents and usually gives cleaner topics on short texts like
    reviews.

READING THE RESULT
    The model only outputs numbered topics. Their meaning is read from their
    keywords (e.g. "game, play, fun, graphics" = video games). The short names
    shown in the app were written by people from those keywords - see
    TOPIC_LABELS in app/config.py.

REPRODUCIBILITY
    random_state=42 and init="nndsvda" make NMF deterministic. Re-running this
    configuration with scikit-learn 1.8 reproduced the stored topic of 99.9% of
    reviews; topic_keywords.csv comes from that re-run.

RUN (from the project root)
    python pipeline/08_topic_extraction.py
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_sentiment.csv")

# Combine title + review
df["text"] = (
    df["title"].fillna("") + " " +
    df["review"].fillna("")
)

# Basic cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("Cleaning reviews...")

df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF
print("Creating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    min_df=5,
    max_df=0.90,
    ngram_range=(1, 2)
)

tfidf = vectorizer.fit_transform(df["clean_text"])

print("TF-IDF shape:", tfidf.shape)

# NMF topic extraction
print("\nExtracting topics...")

NUM_TOPICS = 10

nmf = NMF(
    n_components=NUM_TOPICS,
    random_state=42,
    init="nndsvda",
    max_iter=200
)

# fit() learns W and H. nmf.components_ is H: shape (10 topics, 10000 terms).
nmf.fit(tfidf)

feature_names = vectorizer.get_feature_names_out()

# Display topics
print("\n" + "=" * 70)
print("DISCOVERED TOPICS")
print("=" * 70)

topic_keywords = []

for topic_idx, topic in enumerate(nmf.components_):

    # The 10 terms with the highest weight in this topic, strongest first.
    top_indices = topic.argsort()[-10:][::-1]

    words = [
        feature_names[i]
        for i in top_indices
    ]

    print(
        f"\nTopic {topic_idx + 1}: "
        + ", ".join(words)
    )

    topic_keywords.append({
        "topic": topic_idx + 1,
        "keywords": ", ".join(words)
    })

# Assign dominant topic to each review
print("\nAssigning topics to reviews...")

# transform() returns W: shape (100000 reviews, 10 topics).
topic_scores = nmf.transform(tfidf)

# argmax picks each review's strongest topic; +1 numbers topics 1..10.
df["topic"] = topic_scores.argmax(axis=1) + 1

# Topic distribution
print("\n" + "=" * 70)
print("TOPIC DISTRIBUTION")
print("=" * 70)

print(df["topic"].value_counts().sort_index())

# Save
df[
    ["title", "review", "vader_sentiment",
     "vader_score", "topic"]
].to_csv(
    "amazon_reviews_topics.csv",
    index=False
)

# Save each topic's keywords so the app can show what a topic is about.
pd.DataFrame(topic_keywords).to_csv(
    "topic_keywords.csv",
    index=False
)

print("\nSaved:")
print("amazon_reviews_topics.csv")
print("topic_keywords.csv")

print("\nTopic extraction completed!")
