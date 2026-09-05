"""HTTP API and static host for the Amazon Review NLP Intelligence System."""

import os
import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__, config, data, semantic, sentiment, summarize

api = FastAPI(title="Review Intelligence", version=__version__)

# Long enough, and multi-sentence, to make the summariser actually run.
_ENGINE_PROBE = " ".join(f"Probe sentence number {n} exercises the ranking graph." for n in range(6))


# ------------------------------------------------------------------
# Request bodies
# ------------------------------------------------------------------

class TextPayload(BaseModel):
    text: str = Field(min_length=1, max_length=config.MAX_INPUT_CHARS)


class SummaryPayload(TextPayload):
    sentences: int = Field(default=3, ge=1, le=8)


class SearchPayload(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=config.DEFAULT_RESULTS, ge=1, le=config.MAX_RESULTS)


class NeighbourPayload(BaseModel):
    index: int = Field(ge=0)
    top_k: int = Field(default=config.DEFAULT_RESULTS, ge=1, le=config.MAX_RESULTS)


# ------------------------------------------------------------------
# Corpus overview
# ------------------------------------------------------------------

def _distribution(series):
    counts = series.astype(str).str.strip().str.lower().value_counts()
    return {key: int(counts.get(key, 0)) for key in ("positive", "negative", "neutral")}


@api.get("/api/status")
def status():
    """Which analytical engines are live on this install."""
    matrix = data.embeddings()
    return {
        "version": __version__,
        "corpus": data.corpus_size(),
        "engines": {
            "sentiment": sentiment.engine_name(),
            "search": semantic.search_mode(),
            "similarity": "embeddings" if matrix is not None else "unavailable",
            "summarizer": summarize.summarize(_ENGINE_PROBE, 2)["engine"],
        },
    }


@api.get("/api/overview")
def overview():
    labels = data.sentiment_labels()
    summary = data.aspect_summary()
    topics = data.topics()

    total = int(len(labels)) if labels is not None else 0
    labelled = _distribution(labels["sentiment"]) if (
        labels is not None and "sentiment" in labels.columns) else None
    predicted = _distribution(labels["vader_sentiment"]) if (
        labels is not None and "vader_sentiment" in labels.columns) else None

    agreement = None
    if labels is not None and {"sentiment", "vader_sentiment"} <= set(labels.columns):
        left = labels["sentiment"].astype(str).str.lower()
        right = labels["vader_sentiment"].astype(str).str.lower()
        agreement = round(float((left == right).mean()), 4)

    aspects = []
    if summary is not None:
        frame = summary.copy()
        frame["total"] = frame[["negative", "neutral", "positive"]].sum(axis=1)
        frame = frame.sort_values("total", ascending=False)
        aspects = [
            {
                "aspect": str(row["aspect"]),
                "negative": int(row["negative"]),
                "neutral": int(row["neutral"]),
                "positive": int(row["positive"]),
                "total": int(row["total"]),
                "positive_share": round(float(row["positive"]) / float(row["total"]), 4)
                if row["total"] else 0.0,
            }
            for _, row in frame.iterrows()
        ]

    topic_counts = []
    if topics is not None and "topic" in topics.columns:
        counts = topics["topic"].value_counts().sort_index()
        topic_counts = [{"topic": int(k), "count": int(v)} for k, v in counts.items()]

    return {
        "total_reviews": total,
        "indexed_reviews": data.corpus_size(),
        "labelled": labelled,
        "predicted": predicted,
        "agreement": agreement,
        "aspects": aspects,
        "topics": topic_counts,
        "plates": _plates(),
    }


def _plates():
    """Plate manifest, each src stamped with the file mtime to defeat caching."""
    manifest = []
    for pid, title, filename, caption in config.PLATES:
        path = os.path.join(config.VISUALS_DIR, filename)
        if not os.path.exists(path):
            continue
        manifest.append({
            "id": pid,
            "title": title,
            "src": f"/visuals/{filename}?v={int(os.path.getmtime(path))}",
            "caption": caption,
        })
    return manifest


# ------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------

@api.post("/api/sentiment")
def analyze_sentiment(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a review to analyse.")
    return sentiment.analyze(text)


@api.post("/api/summarize")
def build_summary(payload: SummaryPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a review to summarise.")
    result = summarize.summarize(text, payload.sentences)
    result["original"] = text
    return result


@api.get("/api/aspects")
def aspect_table():
    summary = data.aspect_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="aspect_sentiment_summary.csv not found.")

    rows = []
    for _, row in summary.iterrows():
        negative = int(row["negative"])
        neutral = int(row["neutral"])
        positive = int(row["positive"])
        total = negative + neutral + positive or 1
        rows.append({
            "aspect": str(row["aspect"]),
            "negative": negative,
            "neutral": neutral,
            "positive": positive,
            "total": negative + neutral + positive,
            "positive_share": round(positive / total, 4),
            "negative_share": round(negative / total, 4),
            "neutral_share": round(neutral / total, 4),
            "net": round((positive - negative) / total, 4),
        })
    rows.sort(key=lambda item: -item["total"])
    return {"rows": rows}


@api.get("/api/aspects/{aspect}/reviews")
def aspect_reviews(aspect: str, limit: int = Query(default=6, ge=1, le=30)):
    frame = data.aspects()
    if frame is None:
        raise HTTPException(status_code=404, detail="amazon_review_aspects.csv not found.")

    column = data.find_column(frame, ["aspect"])
    text_column = data.review_column(frame.drop(columns=[column])) if column else None
    if column is None or text_column is None:
        raise HTTPException(status_code=422, detail="Aspect columns could not be identified.")

    matched = frame[frame[column].astype(str).str.lower() == aspect.lower()]
    if matched.empty:
        return {"aspect": aspect, "reviews": []}

    sentiment_column = data.find_column(matched, ["sentiment"])
    sample = matched.head(400).sample(n=min(limit, len(matched)), random_state=7)
    return {
        "aspect": aspect,
        "reviews": [
            {
                "text": str(row[text_column]),
                "sentiment": str(row[sentiment_column]).lower() if sentiment_column else "",
            }
            for _, row in sample.iterrows()
        ],
    }


@api.get("/api/topics")
def topic_list():
    frame = data.topics()
    if frame is None:
        raise HTTPException(status_code=404, detail="amazon_reviews_topics.csv not found.")

    column = data.find_column(frame, ["topic"])
    if column is None:
        raise HTTPException(status_code=422, detail="Topic column could not be identified.")

    counts = frame[column].value_counts().sort_index()
    sentiment_column = data.find_column(frame, ["sentiment"])

    topics = []
    for key, count in counts.items():
        entry = {"topic": int(key), "count": int(count)}
        if sentiment_column is not None:
            subset = frame.loc[frame[column] == key, sentiment_column]
            entry["sentiment"] = _distribution(subset)
        topics.append(entry)
    return {"topics": topics, "total": int(len(frame))}


@api.get("/api/topics/{topic}/reviews")
def topic_reviews(topic: int, limit: int = Query(default=8, ge=1, le=30)):
    frame = data.topics()
    if frame is None:
        raise HTTPException(status_code=404, detail="amazon_reviews_topics.csv not found.")

    column = data.find_column(frame, ["topic"])
    text_column = data.review_column(frame)
    if column is None or text_column is None:
        raise HTTPException(status_code=422, detail="Topic columns could not be identified.")

    matched = frame[frame[column] == topic]
    if matched.empty:
        return {"topic": topic, "count": 0, "reviews": []}

    sentiment_column = data.find_column(matched, ["sentiment"])
    sample = matched.head(500).sample(n=min(limit, len(matched)), random_state=11)
    return {
        "topic": topic,
        "count": int(len(matched)),
        "reviews": [
            {
                "title": str(row.get("title", "")),
                "text": str(row[text_column]),
                "sentiment": str(row[sentiment_column]).lower() if sentiment_column else "",
            }
            for _, row in sample.iterrows()
        ],
    }


@api.post("/api/search")
def search(payload: SearchPayload):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a search query.")
    result = semantic.search(query, payload.top_k)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    result["query"] = query
    return result


@api.get("/api/reviews/{index}")
def review(index: int):
    record = data.review_at(index)
    if record is None:
        raise HTTPException(status_code=404, detail="Review index out of range.")
    record["corpus"] = data.corpus_size()
    return record


@api.get("/api/reviews/random/pick")
def random_review():
    size = data.corpus_size()
    if not size:
        raise HTTPException(status_code=404, detail="Corpus unavailable.")
    return review(random.randrange(size))


@api.post("/api/similar")
def similar(payload: NeighbourPayload):
    result = semantic.neighbours(payload.index, payload.top_k)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    result["source"] = data.review_at(payload.index)
    return result


# ------------------------------------------------------------------
# Static assets
# ------------------------------------------------------------------

@api.exception_handler(404)
def not_found(request, exc):
    if request.url.path.startswith(("/api/", "/visuals/")):
        return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
    return FileResponse(os.path.join(config.WEB_DIR, "index.html"))


if os.path.isdir(config.VISUALS_DIR):
    api.mount("/visuals", StaticFiles(directory=config.VISUALS_DIR), name="visuals")

api.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
