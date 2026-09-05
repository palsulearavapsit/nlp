"""Meaning-based retrieval over the review corpus.

Nearest-neighbour lookups run on the pre-computed embedding matrix and need no
model at all. Free-text search has to encode the query, so it uses
``sentence-transformers`` when present and degrades to a transparent lexical
ranker when it is not - the response always states which mode answered.
"""

import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from app import config, data

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@lru_cache(maxsize=1)
def _model():
    """The sentence encoder, or ``None`` when the package is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception:
        return None


def search_mode():
    return "semantic" if _model() is not None else "lexical"


def cosine_similarity(query_vector, matrix):
    query_vector = np.asarray(query_vector, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return matrix @ query_vector / (norms + 1e-10)


@lru_cache(maxsize=1)
def _lexical_index():
    """IDF-weighted token bags for the fallback ranker."""
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
            continue
        dot = sum(query_bag[t] * bag[t] * idf.get(t, 0.0) ** 2 for t in overlap)
        scores[i] = dot / (query_norm * norms[i])
    return scores


def _rank(scores, top_k, exclude=None):
    order = np.argsort(scores)[::-1]
    picked = []
    for index in order:
        if exclude is not None and int(index) == int(exclude):
            continue
        picked.append(int(index))
        if len(picked) == top_k:
            break
    return picked


def _hydrate(indices, scores):
    results = []
    for rank, index in enumerate(indices, start=1):
        record = data.review_at(index)
        if record is None:
            continue
        record["rank"] = rank
        record["score"] = float(scores[index])
        results.append(record)
    return results


def search(query, top_k=config.DEFAULT_RESULTS):
    """Rank reviews against a free-text query."""
    matrix = data.embeddings()
    model = _model()

    if model is not None and matrix is not None:
        vector = model.encode(query, convert_to_numpy=True)
        scores = cosine_similarity(vector, matrix)
        mode = "semantic"
    else:
        scores = _lexical_scores(query)
        mode = "lexical"

    if scores is None:
        return {"mode": mode, "results": [], "error": "corpus unavailable"}

    limit = max(0, min(top_k, len(scores)))
    return {"mode": mode, "results": _hydrate(_rank(scores, limit), scores)}


def neighbours(index, top_k=config.DEFAULT_RESULTS):
    """Reviews closest in embedding space to the review at ``index``."""
    matrix = data.embeddings()
    if matrix is None:
        return {"mode": "semantic", "results": [], "error": "embeddings unavailable"}
    if index < 0 or index >= len(matrix):
        return {"mode": "semantic", "results": [], "error": "index out of range"}

    scores = cosine_similarity(matrix[int(index)], matrix)
    limit = max(0, min(top_k, len(scores) - 1))
    return {"mode": "semantic", "results": _hydrate(_rank(scores, limit, index), scores)}
