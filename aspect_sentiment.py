import pandas as pd
import re
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
print("Loading dataset...")
df = pd.read_csv("amazon_reviews_sentiment.csv")

# VADER
analyzer = SentimentIntensityAnalyzer()

# Common product-review aspects
ASPECTS = [
    "price",
    "quality",
    "battery",
    "performance",
    "design",
    "size",
    "sound",
    "camera",
    "screen",
    "display",
    "comfort",
    "durability",
    "material",
    "packaging",
    "shipping",
    "delivery",
    "service",
    "customer service",
    "seller",
    "product",
    "book",
    "story",
    "movie",
    "music",
    "audio",
    "video",
    "software",
    "value"
]

def find_aspects(text):
    text = str(text).lower()
    found = []

    for aspect in ASPECTS:
        if re.search(r"\b" + re.escape(aspect) + r"\b", text):
            found.append(aspect)

    return found


def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


print("Extracting aspects...")

aspect_results = []

for _, row in df.iterrows():

    text = str(row["title"]) + " " + str(row["review"])

    aspects = find_aspects(text)

    for aspect in aspects:
        sentiment = get_sentiment(text)

        aspect_results.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "review": text
        })


aspect_df = pd.DataFrame(aspect_results)

print("\n" + "=" * 60)
print("ASPECT FREQUENCY")
print("=" * 60)

print(
    aspect_df["aspect"]
    .value_counts()
    .head(20)
)


print("\n" + "=" * 60)
print("ASPECT SENTIMENT")
print("=" * 60)

summary = (
    aspect_df
    .groupby(["aspect", "sentiment"])
    .size()
    .unstack(fill_value=0)
)

print(summary)


# Save results
aspect_df.to_csv(
    "amazon_review_aspects.csv",
    index=False
)

summary.to_csv(
    "aspect_sentiment_summary.csv"
)

print("\nSaved:")
print("amazon_review_aspects.csv")
print("aspect_sentiment_summary.csv")

print("\nAspect-based sentiment analysis completed!")