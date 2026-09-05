import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading reviews...")

# Load the same 5,000 reviews used to create embeddings
df = pd.read_csv("amazon_reviews_summarized.csv")

df["review"] = df["review"].fillna("")

reviews = df["review"].tolist()

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading review embeddings...")

embeddings = pd.read_csv("review_embeddings.csv").values

print("Embeddings shape:", embeddings.shape)


def semantic_search(query, top_k=5):

    print("\nSearching for:")
    print(query)

    # Convert search query into embedding
    query_embedding = model.encode([query])

    # Calculate similarity
    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get highest scores
    top_indices = scores.argsort()[-top_k:][::-1]

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        result = {
            "rank": rank,
            "similarity": scores[idx],
            "review": reviews[idx]
        }

        results.append(result)

        print("\n" + "-" * 70)
        print(f"Rank: {rank}")
        print(f"Similarity: {scores[idx]:.4f}")
        print("Review:")
        print(reviews[idx][:700])

    return results


# Example searches
queries = [
    "customers complaining about poor product quality",
    "people who love the music",
    "problems with buying a product"
]

for query in queries:
    semantic_search(query, top_k=5)


print("\n" + "=" * 70)
print("SEMANTIC SEARCH COMPLETED!")
print("=" * 70)