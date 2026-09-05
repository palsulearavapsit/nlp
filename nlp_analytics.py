import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import os


# ============================================================
# VISUALS FOLDER
# ============================================================

VISUALS_DIR = "visuals"

os.makedirs(VISUALS_DIR, exist_ok=True)

print("Visuals will be saved in:")
print(os.path.abspath(VISUALS_DIR))


# ============================================================
# LOAD NLP RESULTS
# ============================================================

print("\nLoading NLP results...")

sentiment_df = pd.read_csv(
    "amazon_reviews_sentiment.csv"
)

topic_df = pd.read_csv(
    "amazon_reviews_topics.csv"
)

aspect_df = pd.read_csv(
    "amazon_review_aspects.csv"
)

print("Datasets loaded successfully.")


# ============================================================
# 1. SENTIMENT DISTRIBUTION
# ============================================================

print("\nCreating sentiment distribution...")

sentiment_counts = (
    sentiment_df["vader_sentiment"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

sentiment_counts.plot(
    kind="bar"
)

plt.title(
    "Amazon Review Sentiment Distribution"
)

plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALS_DIR,
        "sentiment_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. TOPIC DISTRIBUTION
# ============================================================

print("Creating topic distribution...")

topic_counts = (
    topic_df["topic"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 5))

topic_counts.plot(
    kind="bar"
)

plt.title(
    "Amazon Review Topic Distribution"
)

plt.xlabel("Topic")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALS_DIR,
        "topic_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. TOP REVIEW ASPECTS
# ============================================================

print("Creating aspect frequency chart...")

aspect_counts = (
    aspect_df["aspect"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10, 6))

aspect_counts.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 15 Review Aspects"
)

plt.xlabel("Number of Mentions")
plt.ylabel("Aspect")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALS_DIR,
        "aspect_frequency.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. ASPECT SENTIMENT
# ============================================================

print("Creating aspect sentiment chart...")

aspect_sentiment = pd.crosstab(
    aspect_df["aspect"],
    aspect_df["sentiment"]
)

# Select top 15 aspects
top_aspects = (
    aspect_df["aspect"]
    .value_counts()
    .head(15)
    .index
)

aspect_sentiment = (
    aspect_sentiment
    .loc[
        aspect_sentiment.index.intersection(
            top_aspects
        )
    ]
)

# Make sure all sentiment columns exist
for sentiment in [
    "positive",
    "negative",
    "neutral"
]:

    if sentiment not in aspect_sentiment.columns:

        aspect_sentiment[sentiment] = 0


aspect_sentiment = aspect_sentiment[
    [
        "positive",
        "negative",
        "neutral"
    ]
]


plt.figure(figsize=(12, 7))

aspect_sentiment.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Sentiment by Review Aspect"
)

plt.xlabel("Aspect")
plt.ylabel("Number of Mentions")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALS_DIR,
        "aspect_sentiment.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. REVIEW LENGTH DISTRIBUTION
# ============================================================

print("Creating review length analysis...")

reviews = (
    sentiment_df["review"]
    .fillna("")
)

review_lengths = reviews.apply(
    lambda x: len(
        str(x).split()
    )
)

print("\nReview Length Statistics:")

print(
    review_lengths.describe()
)


plt.figure(figsize=(9, 5))

plt.hist(
    review_lengths,
    bins=50
)

plt.title(
    "Amazon Review Length Distribution"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Number of Reviews"
)

# Ignore extreme outliers
plt.xlim(
    0,
    review_lengths.quantile(0.99)
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALS_DIR,
        "review_length_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 6. TOP WORDS
# ============================================================

print("\nFinding most common words...")

stopwords = {
    "the",
    "and",
    "a",
    "to",
    "of",
    "in",
    "is",
    "it",
    "this",
    "for",
    "that",
    "was",
    "with",
    "on",
    "as",
    "but",
    "be",
    "are",
    "have",
    "had",
    "i",
    "my",
    "me",
    "we",
    "you",
    "they",
    "he",
    "she",
    "an",
    "or",
    "from",
    "at",
    "so",
    "very",
    "not",
    "its",
    "just"
}


word_counter = Counter()


for text in reviews:

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        str(text).lower()
    )

    words = [
        word
        for word in words
        if word not in stopwords
    ]

    word_counter.update(words)


top_words = (
    word_counter
    .most_common(20)
)


print("\nTop 20 Words:")

for word, count in top_words:

    print(
        f"{word}: {count}"
    )


words_df = pd.DataFrame(
    top_words,
    columns=[
        "word",
        "count"
    ]
)


plt.figure(figsize=(10, 7))

plt.barh(
    words_df["word"][::-1],
    words_df["count"][::-1]
)

plt.title(
    "Top 20 Words in Amazon Reviews"
)

plt.xlabel(
    "Frequency"
)

plt.ylabel(
    "Word"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALS_DIR,
        "top_words.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "NLP ANALYTICS COMPLETED"
)

print("=" * 70)

print(
    "\nAll visualization files saved in:"
)

print(
    os.path.abspath(VISUALS_DIR)
)

print("\nGenerated files:")

print(
    "1. visuals/sentiment_distribution.png"
)

print(
    "2. visuals/topic_distribution.png"
)

print(
    "3. visuals/aspect_frequency.png"
)

print(
    "4. visuals/aspect_sentiment.png"
)

print(
    "5. visuals/review_length_distribution.png"
)

print(
    "6. visuals/top_words.png"
)

print(
    "\nAll analytics completed successfully!"
)