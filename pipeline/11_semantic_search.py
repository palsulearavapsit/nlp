"""
PIPELINE STAGE 11 - SEMANTIC SEARCH
===================================

WHAT THIS STAGE DOES
    Demonstrates searching the 5,000 reviews by MEANING: a free-text query is
    embedded with the same model as the reviews and the closest reviews are
    returned. Runs three example queries and prints the top 5 for each.

    DEMONSTRATION STAGE: it writes no file. The live version of exactly this
    logic is app/semantic.py -> search(), used by the "Retrieval" page.

INPUT
    amazon_reviews_summarized.csv    (from stage 09) - the review texts
    review_embeddings.csv            (from stage 10) - their embeddings
OUTPUT
    console only

NLP CONCEPT - SEMANTIC SEARCH vs KEYWORD SEARCH
    Keyword search returns documents that contain the query's WORDS.
    Semantic search returns documents whose MEANING is closest to the query's.
    A search for "poor product quality" can return a review that says
    "fell apart in two days" although no word matches.

HOW IT WORKS
    1. Encode the query with all-MiniLM-L6-v2 -> one 384-number vector. It MUST
       be the same model that produced review_embeddings.csv, otherwise the
       vectors live in different spaces and cannot be compared.
    2. Cosine similarity between the query vector and all 5,000 review vectors
       -> 5,000 scores.
    3. Sort by score and return the top_k reviews.

RUN (from the project root)
    python pipeline/11_semantic_search.py
"""

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

# Row i of this matrix is the embedding of reviews[i].
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
