"""
PIPELINE STAGE 06 - SENTIMENT ANALYSIS (VADER)
==============================================

WHAT THIS STAGE DOES
    Gives every one of the 100,000 reviews a predicted sentiment label
    (positive / negative / neutral) and a score, then measures how often that
    prediction agrees with the human label that came with the dataset.

INPUT
    amazon_reviews_100k.csv          (from stage 01)
OUTPUT
    amazon_reviews_sentiment.csv     adds: text, vader_sentiment, vader_score

    The app reads this file for the corpus overview page (label counts and the
    "model agreement" figure) - see app/data.py -> sentiment_labels().

NLP CONCEPT - LEXICON-BASED SENTIMENT ANALYSIS WITH VADER
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
    sentiment model. It has a dictionary of about 7,500 words, each rated for
    how positive or negative it is (e.g. "good" +1.9, "horrible" -2.5). It sums
    the ratings of the words in a text and adjusts them with rules for:
        * negation      ("not good" flips the sign)
        * intensifiers  ("very good" is stronger than "good")
        * punctuation   ("good!!!" is stronger than "good")
        * capitals      ("GOOD" is stronger than "good")
        * "but"         (the part after "but" counts more)
    It needs no training data, which is why it can label 100k reviews directly.

    VADER's main output is the COMPOUND score: the summed ratings squashed into
    the range -1 (most negative) to +1 (most positive).

HOW THE LABEL IS CHOSEN (standard VADER thresholds)
    compound >=  0.05  -> positive
    compound <= -0.05  -> negative
    otherwise          -> neutral

RESULT ON THIS DATASET
    VADER agreed with the human label on 70.45% of reviews. Most mistakes are
    human-negative reviews that VADER called positive (24,180 of the 50,000),
    because product complaints often use polite or positive-sounding words
    ("I really wanted to like it..." scores +0.41). This weakness is why the
    live app adds a product-review rule layer on top of VADER - see
    app/sentiment.py. Stage 12 measures that hybrid engine at 76.27%.
    Because the human labels have no "neutral", every review VADER calls
    neutral also counts as a disagreement.

RUN (from the project root)
    python pipeline/06_sentiment_analysis.py
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_100k.csv")

# Combine title + review: the headline often carries the strongest opinion
# ("Total waste of money"), so both parts are analysed together.
df["text"] = (
    df["title"].fillna("") + " " +
    df["review"].fillna("")
)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    """Return (label, compound score) for one review."""
    # polarity_scores() returns {"neg": .., "neu": .., "pos": .., "compound": ..}
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
# "Agreement" = fraction of reviews where VADER's label equals the human label.
# It is effectively VADER's accuracy on this dataset.

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
