"""
PIPELINE STAGE 12 - EVALUATION: VADER alone vs the hybrid engine
================================================================

WHAT THIS STAGE DOES
    Measures how well each sentiment method agrees with the human labels on
    all 100,000 reviews:
        1. VADER alone        - the labels already stored by stage 06
        2. The hybrid engine  - VADER + product-review rules, the live engine
                                behind the Verdict page (app/sentiment.py)
    Prints accuracy, a confusion matrix, and precision / recall / F1 for each.

    Added after the original pipeline, to answer "is the hybrid engine
    actually better?". It writes no file.

INPUT   amazon_reviews_sentiment.csv   (from stage 06)
OUTPUT  console only

EVALUATION TERMS
    Accuracy   share of reviews whose predicted label equals the human label.
               (The app calls this "model agreement".)
    Confusion matrix
               rows = human label, columns = predicted label. The diagonal is
               correct; everything else is a specific kind of mistake.
    Precision  of the reviews predicted "negative", how many really are.
    Recall     of the reviews that really are negative, how many were found.
    F1         the harmonic mean of precision and recall (one balanced score).
    Note: the human labels have no "neutral" class, so every "neutral"
    prediction counts as a mistake for both methods.

RESULT (all 100,000 reviews, measured with vaderSentiment 3.3.2)
                        accuracy   negative F1   positive F1
    VADER alone          70.45%       0.626         0.768
    Hybrid engine        76.27%       0.727         0.803
    The gain comes from complaints: VADER called 24,180 human-negative reviews
    positive; the hybrid engine calls 17,833 of them positive (26% fewer).
    (The "neutral" row of the report is all zeros because no human label is
    neutral; ignore it and the "macro avg" line it drags down.)

RUN (from the project root; needs the app's requirements + scikit-learn)
    python pipeline/12_evaluate_sentiment.py
"""

import os
import sys

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Make the app package importable when this file is run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sentiment  # noqa: E402

LABELS = ["negative", "neutral", "positive"]

print("Loading dataset...")

# "text" = title + review, exactly what VADER analysed in stage 06, so both
# methods see the same input.
df = pd.read_csv(
    "amazon_reviews_sentiment.csv",
    usecols=["text", "sentiment", "vader_sentiment"]
)

print(f"Running the hybrid engine on {len(df):,} reviews (takes a few minutes)...")

# app/sentiment.py returns "Positive" / "Negative" / "Neutral"; lowercase it to
# match the dataset's labels.
df["hybrid_sentiment"] = df["text"].astype(str).map(
    lambda text: sentiment.analyze(text)["sentiment"].lower()
)

if sentiment.engine_name() != "vader+rules":
    print("\nWARNING: vaderSentiment is not installed, so the 'hybrid' column is "
          "the rules-only fallback, not the real hybrid engine.")

for name, column in [("VADER ALONE (stage 06)", "vader_sentiment"),
                     ("HYBRID ENGINE (app/sentiment.py)", "hybrid_sentiment")]:
    accuracy = (df["sentiment"] == df[column]).mean() * 100

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy (agreement with human label): {accuracy:.2f}%")

    print("\nConfusion matrix (rows = human, columns = predicted):")
    matrix = confusion_matrix(df["sentiment"], df[column], labels=LABELS)
    print(pd.DataFrame(matrix, index=[f"human {l}" for l in LABELS],
                       columns=[f"pred {l}" for l in LABELS]))

    print("\nPer-class precision / recall / F1:")
    print(classification_report(df["sentiment"], df[column], labels=LABELS,
                                digits=3, zero_division=0))

print("\nEvaluation completed!")
