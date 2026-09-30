"""Lazy, cached access to the corpus artefacts produced by the NLP pipeline.

ROLE IN THE PROJECT
    The only module that reads data files from disk. Every other module asks
    this one for data (summaries(), topics(), embeddings(), ...) instead of
    opening files itself.

WHY "LAZY" AND "CACHED"?
    The CSVs total about 260 MB. Loading them all at start-up would make the
    server slow to start, so each file is loaded the FIRST time a feature needs
    it (lazy) and then kept in memory for every later request (cached).
    Note: the engine-status box in the left rail (GET /api/status) runs on
    every first page view, and it loads the embeddings and the 5,000-review
    summaries - so those two load as soon as the page opens.

    The embeddings are also cached ON DISK as .cache/review_embeddings.npy. If
    pipeline stage 10 is ever re-run, delete the .cache folder so the new
    embeddings are picked up.

CONNECTED TO
    config.py   - file paths
    server.py   - overview / aspect / topic / review endpoints
    semantic.py - review texts and the embedding matrix
    plates.py   - chart data
"""

import os
import threading

import numpy as np
import pandas as pd

from app import config

_lock = threading.Lock()
_cache = {}  # name -> loaded DataFrame / array (or None if the file is missing)

# Different CSVs name their text column differently; the first match wins.
REVIEW_COLUMN_CANDIDATES = ["review", "review_text", "text", "combined_text", "content"]


def _load_once(key, loader):
    """Run ``loader()`` the first time ``key`` is requested, then reuse the result.

    The web server handles requests on several threads. The lock makes a second
    request wait for the first load to finish instead of loading the same file
    twice.
    """
    if key in _cache:
        return _cache[key]
    with _lock:
        if key not in _cache:
            _cache[key] = loader()
    return _cache[key]


def _read_csv(path, **kwargs):
    """Read a CSV, or return None when the file is missing (the feature then degrades)."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)


def find_column(df, possible_names):
    """First column whose name contains any of ``possible_names``."""
    if df is None:
        return None
    for column in df.columns:
        lowered = str(column).lower()
        for name in possible_names:
            if name.lower() in lowered:
                return column
    return None


def review_column(df):
    """The column holding the review text (falls back to the first text column)."""
    if df is None:
        return None
    column = find_column(df, REVIEW_COLUMN_CANDIDATES)
    if column:
        return column
    text_columns = df.select_dtypes(include="object").columns
    return text_columns[0] if len(text_columns) else None


# ---------------------------------------------------------------------------
# One accessor per pipeline output
# ---------------------------------------------------------------------------

def summaries():
    """The 5,000-review retrieval corpus with TextRank summaries (pipeline stage 09)."""
    return _load_once("summaries", lambda: _read_csv(config.SUMMARY_FILE))


def topics():
    """All 100k reviews with their NMF topic number 1..10 (pipeline stage 08)."""
    return _load_once("topics", lambda: _read_csv(config.TOPIC_FILE))


def topic_keywords():
    """{topic number: [its top NMF keywords]} (pipeline stage 08), or {} if missing."""

    def load():
        frame = _read_csv(config.TOPIC_KEYWORDS_FILE)
        if frame is None:
            return {}
        return {int(row["topic"]): [word.strip() for word in str(row["keywords"]).split(",")]
                for _, row in frame.iterrows()}

    return _load_once("topic_keywords", load)


def aspects():
    """One row per (aspect, review) pair (pipeline stage 07)."""
    return _load_once("aspects", lambda: _read_csv(config.ASPECT_FILE))


def aspect_summary():
    """28 aspects with negative / neutral / positive counts (pipeline stage 07)."""
    return _load_once("aspect_summary", lambda: _read_csv(config.ASPECT_SUMMARY_FILE))


def sentiment_labels():
    """Only the label columns of the 100k sentiment file - the text is 90MB.

    sentiment        the human label shipped with the dataset (positive / negative)
    vader_sentiment  what VADER predicted in pipeline stage 06
    vader_score      VADER's compound score, -1 .. +1
    """

    def load():
        if not os.path.exists(config.SENTIMENT_FILE):
            return None
        header = pd.read_csv(config.SENTIMENT_FILE, nrows=0)
        wanted = [c for c in ("sentiment", "vader_sentiment", "vader_score")
                  if c in header.columns]
        if not wanted:
            return None
        return pd.read_csv(config.SENTIMENT_FILE, usecols=wanted)

    return _load_once("sentiment_labels", load)


def embeddings():
    """Review embedding matrix, cached to .npy so restarts are instant.

    Shape (5000, 384): row i is the MiniLM sentence embedding of review i of
    summaries(). That shared row order is what lets semantic.py turn "row 42
    scored highest" back into "review 42".
    """

    def load():
        if os.path.exists(config.EMBEDDINGS_CACHE):
            return np.load(config.EMBEDDINGS_CACHE)
        if not os.path.exists(config.EMBEDDINGS_FILE):
            return None

        # Parsing 1.9 million numbers from text takes seconds; the binary .npy
        # copy saved below loads in milliseconds on every later start.
        frame = pd.read_csv(config.EMBEDDINGS_FILE).select_dtypes(include=[np.number])
        if frame.shape[1] >= config.EMBEDDING_DIM:
            frame = frame.iloc[:, -config.EMBEDDING_DIM:]
        matrix = np.ascontiguousarray(frame.values.astype("float32"))

        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.save(config.EMBEDDINGS_CACHE, matrix)
        return matrix

    return _load_once("embeddings", load)


def review_at(index):
    """A single row of the summarised corpus, normalised for the API."""
    frame = summaries()
    if frame is None or index < 0 or index >= len(frame):
        return None

    row = frame.iloc[int(index)]
    text_column = review_column(frame)
    return {
        "index": int(index),
        "title": _clean(row.get("title")),
        "review": _clean(row.get(text_column)) if text_column else "",
        "summary": _clean(row.get("summary")),
        "sentiment": _clean(row.get("vader_sentiment")),
        "score": _number(row.get("vader_score")),
        "topic": _number(row.get("topic")),
    }


def _clean(value):
    """Missing / NaN cells become "" so the JSON never contains NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _number(value):
    """Numbers become floats; missing or non-numeric cells become None."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def corpus_size():
    """Rows addressable by both the summary table and the embedding matrix."""
    frame = summaries()
    matrix = embeddings()
    sizes = [len(frame) if frame is not None else 0,
             len(matrix) if matrix is not None else 0]
    positive = [s for s in sizes if s]
    return min(positive) if positive else 0
