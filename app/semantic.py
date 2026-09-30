"""Meaning-based retrieval over the review corpus.

Nearest-neighbour lookups run on the pre-computed embedding matrix and need no
model at all. Free-text search has to encode the query, so it uses
``sentence-transformers`` when present and degrades to a transparent lexical
ranker when it is not - the response always states which mode answered.

ROLE IN THE PROJECT
    Powers the "Retrieval" page:
      search box             - POST /api/search   -> search(query)
      "Find similar reviews" - POST /api/similar  -> neighbours(index)

NLP CONCEPT - SENTENCE EMBEDDINGS
    pipeline/10 turned each of the 5,000 corpus reviews into a 384-number
    vector with the all-MiniLM-L6-v2 Transformer. Reviews with similar MEANING
    get vectors that point in similar directions, even with no words in common.

NLP CONCEPT - COSINE SIMILARITY
    Closeness of two vectors = cosine of the angle between them:
        cos(a, b) = (a . b) / (|a| x |b|)       1 = same direction, 0 = unrelated
    See cosine_similarity() below - it scores one vector against all 5,000 rows
    in a single matrix multiplication.

TWO SEARCH MODES
    semantic - the query is embedded with the SAME MiniLM model and compared by
               cosine similarity. Finds reviews by meaning.
    lexical  - fallback when sentence-transformers is not installed. Ranks by
               IDF-weighted word overlap (a TF-IDF-style cosine over word
               counts). Finds reviews that share the query's words, weighting
               rare words more.
    "Find similar" never needs the model: the stored vector of review i is already
    in the matrix.
"""

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from app import config, data

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@lru_cache(maxsize=1)
def _sentence_model():
    """The sentence encoder, or ``None`` when the package is unavailable.

    Loaded once (lru_cache) because loading the Transformer takes a few seconds.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception:
        return None


def search_mode():
    """Return "semantic" (MiniLM available) or "lexical" (fallback)."""
    return "semantic" if _sentence_model() is not None else "lexical"


def cosine_similarity(query_vector, matrix):
    """Cosine similarity of one vector against every row of ``matrix``.

    matrix @ query_vector computes all 5,000 dot products at once; dividing by
    the product of the vector lengths (norms) turns each into a cosine.
    The tiny 1e-10 avoids dividing by zero for an all-zero vector.
    """
    query_vector = np.asarray(query_vector, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return matrix @ query_vector / (norms + 1e-10)


# ---------------------------------------------------------------------------
# Lexical fallback ranker (used only when sentence-transformers is missing)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _lexical_index():
    """IDF-weighted token bags for the fallback ranker.

    Built once over the 5,000 reviews:
      documents - one Counter {word: count} per review (bag of words)
      idf       - inverse document frequency of every word:
                      idf(t) = log(1 + N / (1 + df(t)))
                  rare words get a high weight, common words a low one
      norms     - the length of each review's weighted vector, pre-computed so
                  every query only has to compute dot products
    """
    frame = data.summaries()
    if frame is None:
        return None

    column = data.review_column(frame)
    if column is None:
        return None

    documents = [Counter(_TOKEN_RE.findall(str(text).lower()))
                 for text in frame[column].fillna("")]
    document_frequency = Counter()
    for bag in documents:
        document_frequency.update(bag.keys())

    total = len(documents) or 1
    idf = {term: math.log(1 + total / (1 + count))
           for term, count in document_frequency.items()}
    norms = [math.sqrt(sum((count * idf.get(term, 0.0)) ** 2
                           for term, count in bag.items())) or 1.0
             for bag in documents]
    return documents, idf, norms


def _lexical_scores(query):
    """Cosine similarity between the query and every review, in IDF-weighted word space."""
    index = _lexical_index()
    if index is None:
        return None

    documents, idf, norms = index
    query_bag = Counter(_TOKEN_RE.findall(query.lower()))
    if not query_bag:
        return np.zeros(len(documents), dtype="float32")

    query_norm = math.sqrt(
        sum((count * idf.get(term, 0.0)) ** 2 for term, count in query_bag.items())
    ) or 1.0

    scores = np.zeros(len(documents), dtype="float32")
    for i, bag in enumerate(documents):
        overlap = query_bag.keys() & bag.keys()
        if not overlap:
            continue  # no shared word -> similarity stays 0
        # Dot product of the two weighted vectors: each word's weight is
        # count x idf, so a shared word contributes q_count x d_count x idf^2.
        dot = sum(query_bag[t] * bag[t] * idf.get(t, 0.0) ** 2 for t in overlap)
        scores[i] = dot / (query_norm * norms[i])
    return scores


# ---------------------------------------------------------------------------
# Shared ranking helpers
# ---------------------------------------------------------------------------

def _top_k_indices(scores, top_k, exclude=None):
    """Row numbers of the ``top_k`` highest scores, best first.

    ``exclude`` skips one row - used by neighbours() so a review is not listed
    as its own nearest neighbour (its similarity to itself is always 1.0).
    """
    order = np.argsort(scores)[::-1]  # all rows, highest score first
    picked = []
    for index in order:
        if exclude is not None and int(index) == int(exclude):
            continue
        picked.append(int(index))
        if len(picked) == top_k:
            break
    return picked


def _attach_review_details(indices, scores):
    """Turn row numbers into full result records (title, text, summary, ...).

    Works because row i of the embedding matrix and row i of the summaries CSV
    are the same review (see data.embeddings()).
    """
    results = []
    for rank, index in enumerate(indices, start=1):
        record = data.review_at(index)
        if record is None:
            continue
        record["rank"] = rank
        # Replaces the review's VADER score with its similarity to the query.
        record["score"] = float(scores[index])
        results.append(record)
    return results


# ---------------------------------------------------------------------------
# Public API (called by app/server.py)
# ---------------------------------------------------------------------------

def search(query, top_k=config.DEFAULT_RESULTS):
    """Rank reviews against a free-text query.

    1. Encode the query into the same 384-d space as the reviews (semantic mode).
    2. Score every review with cosine similarity.
    3. Return the top_k reviews with their scores.
    """
    matrix = data.embeddings()
    model = _sentence_model()

    if model is not None and matrix is not None:
        vector = model.encode(query, convert_to_numpy=True)  # shape (384,)
        scores = cosine_similarity(vector, matrix)           # shape (5000,)
        mode = "semantic"
    else:
        scores = _lexical_scores(query)
        mode = "lexical"

    if scores is None:
        return {"mode": mode, "results": [], "error": "corpus unavailable"}

    limit = max(0, min(top_k, len(scores)))
    return {"mode": mode, "results": _attach_review_details(_top_k_indices(scores, limit), scores)}


def neighbours(index, top_k=config.DEFAULT_RESULTS):
    """Reviews closest in embedding space to the review at ``index``.

    The "query" is the stored embedding of review ``index`` itself, so no model
    is needed - this always runs in semantic mode.
    """
    matrix = data.embeddings()
    if matrix is None:
        return {"mode": "semantic", "results": [], "error": "embeddings unavailable"}
    if index < 0 or index >= len(matrix):
        return {"mode": "semantic", "results": [], "error": "index out of range"}

    scores = cosine_similarity(matrix[int(index)], matrix)
    limit = max(0, min(top_k, len(scores) - 1))
    return {"mode": "semantic",
            "results": _attach_review_details(_top_k_indices(scores, limit, index), scores)}
