import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_100k.csv")

# Combine title + review
df["text"] = (
    df["title"].fillna("") + " " +
    df["review"].fillna("")
)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, compound


print("\nRunning VADER sentiment analysis...\n")

# Test first 20 reviews
for i in range(20):

    text = df.loc[i, "text"]

    sentiment, score = get_sentiment(text)

    print("=" * 80)
    print("REVIEW:")
    print(text[:300])

    print("\nDATASET LABEL:", df.loc[i, "sentiment"])
    print("VADER:", sentiment)
    print("COMPOUND SCORE:", round(score, 4))


# =====================================================
# ANALYZE ENTIRE DATASET
# =====================================================

print("\nCalculating sentiment for all reviews...")

results = df["text"].apply(get_sentiment)

df["vader_sentiment"] = results.apply(lambda x: x[0])
df["vader_score"] = results.apply(lambda x: x[1])


# =====================================================
# DISTRIBUTION
# =====================================================

print("\nVADER SENTIMENT DISTRIBUTION")
print(df["vader_sentiment"].value_counts())


# =====================================================
# AGREEMENT WITH DATASET LABEL
# =====================================================

comparison = (
    df["sentiment"] == df["vader_sentiment"]
)

accuracy = comparison.mean() * 100

print("\n" + "=" * 60)
print("VADER vs DATASET LABEL")
print("=" * 60)

print(f"Agreement: {accuracy:.2f}%")


# Save results
df.to_csv("amazon_reviews_sentiment.csv", index=False)

print("\nSaved:")
print("amazon_reviews_sentiment.csv")

print("\nSentiment analysis completed!")