import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_summarized.csv")

# Use a small sample for the demonstration
reviews = df["review"].fillna("").head(5000).tolist()

print("Loading NLP embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating semantic embeddings...")

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

scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

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