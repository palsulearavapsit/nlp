"""Lazy, cached access to the corpus artefacts produced by the NLP pipeline."""

import os
import threading

import numpy as np
import pandas as pd

from app import config

_lock = threading.Lock()
_cache = {}

REVIEW_COLUMN_CANDIDATES = ["review", "review_text", "text", "combined_text", "content"]


def _memo(key, loader):
    """Load ``key`` once; concurrent requests wait rather than duplicating work."""
    if key in _cache:
        return _cache[key]
    with _lock:
        if key not in _cache:
            _cache[key] = loader()
    return _cache[key]


def _read_csv(path, **kwargs):
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
    if df is None:
        return None
    column = find_column(df, REVIEW_COLUMN_CANDIDATES)
    if column:
        return column
    text_columns = df.select_dtypes(include="object").columns
    return text_columns[0] if len(text_columns) else None


def summaries():
    return _memo("summaries", lambda: _read_csv(config.SUMMARY_FILE))


def topics():
    return _memo("topics", lambda: _read_csv(config.TOPIC_FILE))


def aspects():
    return _memo("aspects", lambda: _read_csv(config.ASPECT_FILE))


def aspect_summary():
    return _memo("aspect_summary", lambda: _read_csv(config.ASPECT_SUMMARY_FILE))


def sentiment_labels():
    """Only the label columns of the 100k sentiment file - the text is 90MB."""

    def load():
        if not os.path.exists(config.SENTIMENT_FILE):
            return None
        header = pd.read_csv(config.SENTIMENT_FILE, nrows=0)
        wanted = [c for c in ("sentiment", "vader_sentiment", "vader_score")
                  if c in header.columns]
        if not wanted:
            return None
        return pd.read_csv(config.SENTIMENT_FILE, usecols=wanted)

    return _memo("sentiment_labels", load)


def embeddings():
    """Review embedding matrix, cached to .npy so restarts are instant."""

    def load():
        if os.path.exists(config.EMBEDDINGS_CACHE):
            return np.load(config.EMBEDDINGS_CACHE)
        if not os.path.exists(config.EMBEDDINGS_FILE):
            return None

        frame = pd.read_csv(config.EMBEDDINGS_FILE).select_dtypes(include=[np.number])
        if frame.shape[1] >= config.EMBEDDING_DIM:
            frame = frame.iloc[:, -config.EMBEDDING_DIM:]
        matrix = np.ascontiguousarray(frame.values.astype("float32"))

        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.save(config.EMBEDDINGS_CACHE, matrix)
        return matrix

    return _memo("embeddings", load)


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
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _number(value):
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
