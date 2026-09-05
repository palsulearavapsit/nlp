import pandas as pd

INPUT_FILE = "train.csv"
OUTPUT_FILE = "amazon_reviews_100k.csv"

positive = []
negative = []

chunksize = 100000

print("Reading dataset...")

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

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("\nDONE!")
print(f"Total reviews: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())