"""
PIPELINE STAGE 01 - DATASET PREPARATION
=======================================

WHAT THIS STAGE DOES
    Builds the balanced 100,000-review working dataset that every later stage
    reads.

INPUT
    train.csv - the raw training file of the public "Amazon Review Polarity"
    dataset (Zhang et al., 2015; commonly downloaded from Kaggle). It has about
    3.6 million reviews and NO header row. Its three columns are:
        label  : 1 = negative, 2 = positive
        title  : the review headline
        review : the review body
    It is not included in this repository because it is very large.

OUTPUT
    amazon_reviews_100k.csv   columns: title, review, sentiment

HOW IT WORKS
    1. Read the raw file in chunks of 100,000 rows (too big to load at once).
    2. From each chunk, collect positive rows (label 2) and negative rows
       (label 1) until there are at least 50,000 of each.
    3. Keep exactly 50,000 of each class, so the dataset is balanced (50/50).
    4. Replace the numeric label with a word: 1 -> "negative", 2 -> "positive".
    5. Shuffle the rows (random_state=42 makes the shuffle reproducible).

WHY A BALANCED 100k SUBSET?
    * Balanced: a model that always answers "positive" would only get 50% right,
      so the agreement numbers measured later actually mean something.
    * 100k rows is small enough to process on a laptop, large enough to be
      statistically meaningful.

NOTE ON THE LABELS
    The dataset's authors labelled 1-2 star reviews negative and 4-5 star
    reviews positive and dropped 3-star reviews. So the human labels have NO
    "neutral" class. This matters in stage 06.

RUN (from the project root)
    python pipeline/01_prepare_dataset.py
"""

import pandas as pd

INPUT_FILE = "train.csv"
OUTPUT_FILE = "amazon_reviews_100k.csv"

positive = []
negative = []

chunksize = 100000

print("Reading dataset...")

# The raw CSV has no header, so we name the three columns ourselves.
for chunk in pd.read_csv(
    INPUT_FILE,
    header=None,
    names=["label", "title", "review"],
    chunksize=chunksize
):

    # Collect positive reviews
    pos = chunk[chunk["label"] == 2]
    positive.extend(pos.to_dict("records"))

    # Collect negative reviews
    neg = chunk[chunk["label"] == 1]
    negative.extend(neg.to_dict("records"))

    print(
        f"Collected: {len(positive)} positive | "
        f"{len(negative)} negative"
    )

    # Stop once we have enough
    if len(positive) >= 50000 and len(negative) >= 50000:
        break

# Create balanced dataset
positive = positive[:50000]
negative = negative[:50000]

df = pd.DataFrame(positive + negative)

# Convert labels
df["sentiment"] = df["label"].map({
    1: "negative",
    2: "positive"
})

# Remove original numeric label
df = df.drop(columns=["label"])

# Shuffle - otherwise all 50k positives would come first, then all negatives.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("\nDONE!")
print(f"Total reviews: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
