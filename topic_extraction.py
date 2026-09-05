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

nmf.fit(tfidf)

feature_names = vectorizer.get_feature_names_out()

# Display topics
print("\n" + "=" * 70)
print("DISCOVERED TOPICS")
print("=" * 70)

for topic_idx, topic in enumerate(nmf.components_):

    top_indices = topic.argsort()[-10:][::-1]

    words = [
        feature_names[i]
        for i in top_indices
    ]

    print(
        f"\nTopic {topic_idx + 1}: "
        + ", ".join(words)
    )

# Assign dominant topic to each review
print("\nAssigning topics to reviews...")

topic_scores = nmf.transform(tfidf)

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

print("\nSaved:")
print("amazon_reviews_topics.csv")

print("\nTopic extraction completed!")