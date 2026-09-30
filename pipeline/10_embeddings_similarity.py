"""
PIPELINE STAGE 10 - SENTENCE EMBEDDINGS & SEMANTIC SIMILARITY
=============================================================

WHAT THIS STAGE DOES
    Converts each of the 5,000 retrieval-corpus reviews into a vector of 384
    numbers (an "embedding") that captures its MEANING, saves all the vectors,
    and demonstrates finding the reviews most similar to review 0.

INPUT
    amazon_reviews_summarized.csv    (from stage 09)
OUTPUT
    review_embeddings.csv            5,000 rows x 384 columns (named 0..383).
                                     Row i is the embedding of review i of
                                     amazon_reviews_summarized.csv.

    The app loads this matrix once (app/data.py -> embeddings()) and uses it
    for the "Retrieval" page: search and "Find similar reviews" (app/semantic.py).

NLP CONCEPT - SENTENCE EMBEDDINGS
    An embedding maps a text to a point in a high-dimensional space so that
    texts with similar meaning land close together - even if they share no
    words. "Stopped working after a week" and "died within days" end up near
    each other.

    The model is all-MiniLM-L6-v2 from the sentence-transformers library:
        * a small 6-layer Transformer based on Microsoft's MiniLM (itself a
          compressed, "distilled" BERT-style model), fine-tuned on about
          1 billion sentence pairs to put paraphrases close together;
        * it reads the words in context (attention), averages the token
          vectors ("mean pooling") and outputs one 384-number vector per text;
        * it reads at most 256 word-pieces, so very long reviews are cut off.

NLP CONCEPT - COSINE SIMILARITY
    How close two embeddings are is measured by the cosine of the angle
    between them:
        cos(a, b) = (a . b) / (|a| x |b|)
    1 = pointing the same way (same meaning), 0 = unrelated.
    Only the direction matters, not the length, so a long and a short review
    with the same meaning still score high.

RUN (from the project root)
    python pipeline/10_embeddings_similarity.py
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_summarized.csv")

# Use a small sample for the demonstration
# (the file already has exactly 5,000 rows, so this is all of them).
# Note: only the review body is embedded, not the title.
reviews = df["review"].fillna("").head(5000).tolist()

print("Loading NLP embedding model...")

# Downloads the model (~90 MB) on first use, then loads it from a local cache.
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating semantic embeddings...")

# encode() returns a NumPy array of shape (5000, 384).
embeddings = model.encode(
    reviews,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)

# Compare two example reviews
review_a = reviews[0]
review_b = reviews[1]

similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]

print("\n" + "=" * 70)
print("SEMANTIC SIMILARITY EXAMPLE")
print("=" * 70)

print("\nReview A:")
print(review_a)

print("\nReview B:")
print(review_b)

print(f"\nSimilarity Score: {similarity:.4f}")

# Find reviews similar to the first review
print("\n" + "=" * 70)
print("MOST SIMILAR REVIEWS")
print("=" * 70)

query_embedding = embeddings[0].reshape(1, -1)

# One similarity score per review: shape (5000,).
scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

# Top 6 = review 0 itself (similarity 1.0) + its 5 nearest neighbours.
similar_indices = scores.argsort()[-6:][::-1]

for rank, idx in enumerate(similar_indices, start=1):

    print("\n" + "-" * 60)
    print(f"Rank {rank}")
    print(f"Similarity: {scores[idx]:.4f}")
    print(reviews[idx][:500])

# Save embeddings
embedding_df = pd.DataFrame(
    embeddings
)

embedding_df.to_csv(
    "review_embeddings.csv",
    index=False
)

print("\nSaved:")
print("review_embeddings.csv")

print("\nSemantic similarity completed!")
