"""
PIPELINE STAGE 07 - ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
==========================================================

WHAT THIS STAGE DOES
    Finds WHICH product aspects each review talks about (price, battery,
    shipping, ...) and counts how often each aspect appears in positive,
    negative and neutral reviews.

INPUT
    amazon_reviews_sentiment.csv       (from stage 06)
OUTPUT
    amazon_review_aspects.csv          one row per (aspect, review) pair
                                       columns: aspect, sentiment, review
    aspect_sentiment_summary.csv       one row per aspect (28 rows)
                                       columns: aspect, negative, neutral, positive

    The app reads both files for the "Aspects" page - see app/server.py ->
    aspect_table() and aspect_reviews().

NLP CONCEPT - ASPECT-BASED SENTIMENT ANALYSIS
    Ordinary sentiment analysis gives one label per review. But a review can
    praise the battery and complain about the screen. ABSA asks "what is the
    opinion about EACH aspect?". It has two sub-tasks:
        1. Aspect extraction - which aspects are mentioned?
        2. Aspect sentiment  - what is the opinion about each one?

HOW THIS PROJECT DOES IT
    1. Aspect extraction is LEXICON-BASED: a hand-picked list of 28 common
       product-review aspects (ASPECTS below). A review "mentions" an aspect
       when the word appears as a whole word. \\b...\\b in the regex marks word
       boundaries, so "price" matches "the price is" but not "priceless".
    2. Aspect sentiment: the review's VADER label (same thresholds as stage 06)
       is attached to every aspect found in it.
    3. The pairs are counted with groupby(aspect, sentiment) to build the
       summary table.

KNOWN LIMITATION (be ready to explain this)
    Step 2 uses the sentiment of the WHOLE review, not of the sentence that
    mentions the aspect. "Great battery, awful screen" gives battery and screen
    the same label. A finer version would split the review into sentences and
    score only the sentence(s) that contain each aspect.

RUN (from the project root)
    python pipeline/07_aspect_sentiment.py
"""

import pandas as pd
import re
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
print("Loading dataset...")
df = pd.read_csv("amazon_reviews_sentiment.csv")

# VADER
analyzer = SentimentIntensityAnalyzer()

# Common product-review aspects - the "aspect lexicon".
# Chosen by hand to cover electronics, media (book/movie/music) and buying
# experience (shipping, seller, service).
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
    """Return every aspect from ASPECTS that appears as a whole word in text."""
    text = str(text).lower()
    found = []

    for aspect in ASPECTS:
        # \b = word boundary; re.escape keeps "customer service" literal.
        if re.search(r"\b" + re.escape(aspect) + r"\b", text):
            found.append(aspect)

    return found


def get_sentiment(text):
    """VADER label for the whole text (same thresholds as stage 06)."""
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

    # One output row per aspect found. A review mentioning "price" and
    # "quality" produces two rows, both carrying the review's sentiment
    # (see KNOWN LIMITATION above).
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

# Pivot the (aspect, sentiment) pairs into a table:
#   rows = aspects, columns = negative/neutral/positive, cells = counts.
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
