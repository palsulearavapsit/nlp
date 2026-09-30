"""HTTP API and static host for the Amazon Review NLP Intelligence System.

ROLE IN THE PROJECT
    The bridge between the browser and the NLP code. It does two jobs:
      1. Serves the web page itself (web/index.html, web/assets/*).
      2. Answers the page's /api/... requests with JSON.

HOW ONE REQUEST FLOWS (example: the Verdict page)
    browser: web/assets/app.js  ->  fetch POST /api/sentiment {"text": "..."}
    here:    analyze_sentiment()  validates the body (pydantic), then calls
             app/sentiment.py -> analyze(text)
    here:    FastAPI converts the returned dict to JSON
    browser: app.js renderVerdict() turns the JSON into the result panel

ENDPOINT MAP                                  page (web/)       NLP code
    GET  /api/status                          rail            all engines
    GET  /api/overview                        01 Corpus       pipeline CSVs (data.py)
    GET  /visuals/*.png                       01 Corpus       charts from plates.py
    POST /api/sentiment                       02 Verdict      sentiment.analyze
    GET  /api/aspects                         03 Aspects      aspect CSVs (pipeline 07)
    GET  /api/aspects/{aspect}/reviews        03 Aspects      aspect CSVs (pipeline 07)
    GET  /api/topics                          04 Topics       topic CSVs (pipeline 08)
    GET  /api/topics/{topic}/reviews          04 Topics       topic CSVs (pipeline 08)
    POST /api/search                          05 Retrieval    semantic.search
    POST /api/similar                         05 Retrieval    semantic.neighbours
                                              ("Find similar reviews")
    GET  /api/reviews/{index}                 05              data.review_at (first review shown)
    GET  /api/reviews/random/pick             05, 06          data.review_at ("Random", "load one")
    POST /api/summarize                       06 Condense     summarize.summarize

FastAPI also generates interactive API docs at http://127.0.0.1:8000/docs
"""

import os
import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__, config, data, semantic, sentiment, summarize

api = FastAPI(title="Review Intelligence", version=__version__)

# Long enough, and multi-sentence, to make the summariser actually run - used
# by /api/status to discover which summariser engine is live.
_SUMMARIZER_PROBE_TEXT = " ".join(
    f"Probe sentence number {n} exercises the ranking graph." for n in range(6))


# ------------------------------------------------------------------
# Request bodies
# ------------------------------------------------------------------
# pydantic models validate incoming JSON before our code runs: a missing or
# over-long field is rejected with HTTP 422 and a readable message.

class TextPayload(BaseModel):
    text: str = Field(min_length=1, max_length=config.MAX_INPUT_CHARS)


class SummaryPayload(TextPayload):
    sentences: int = Field(default=3, ge=1, le=8)  # summary length, 1..8 sentences


class SearchPayload(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=config.DEFAULT_RESULTS, ge=1, le=config.MAX_RESULTS)


class NeighbourPayload(BaseModel):
    index: int = Field(ge=0)  # row of the 5,000-review corpus
    top_k: int = Field(default=config.DEFAULT_RESULTS, ge=1, le=config.MAX_RESULTS)


# ------------------------------------------------------------------
# Corpus overview
# ------------------------------------------------------------------

def _polarity_counts(series):
    """Count positive / negative / neutral labels in a column of sentiment labels."""
    counts = series.astype(str).str.strip().str.lower().value_counts()
    return {key: int(counts.get(key, 0)) for key in ("positive", "negative", "neutral")}


def _topic_details(topic):
    """Human label + NMF keywords for one topic number (empty when unknown)."""
    return {
        "label": config.TOPIC_LABELS.get(int(topic), ""),
        "keywords": data.topic_keywords().get(int(topic), []),
    }


@api.get("/api/status")
def status():
    """Which analytical engines are live on this install (shown in the left rail)."""
    matrix = data.embeddings()
    return {
        "version": __version__,
        "corpus": data.corpus_size(),
        "engines": {
            "sentiment": sentiment.engine_name(),
            "search": semantic.search_mode(),
            "similarity": "embeddings" if matrix is not None else "unavailable",
            "summarizer": summarize.summarize(_SUMMARIZER_PROBE_TEXT, 2)["engine"],
        },
    }


@api.get("/api/overview")
def overview():
    """Page 01 Corpus: whole-corpus statistics from the pipeline CSVs, plus the chart list.

    No NLP runs here - it counts and summarises what the offline pipeline
    already computed for all 100,000 reviews.
    """
    labels = data.sentiment_labels()
    summary = data.aspect_summary()
    topics = data.topics()

    total = int(len(labels)) if labels is not None else 0
    # "labelled" = human labels shipped with the dataset (always positive/negative).
    labelled = _polarity_counts(labels["sentiment"]) if (
        labels is not None and "sentiment" in labels.columns) else None
    # "predicted" = what VADER said in pipeline stage 06.
    predicted = _polarity_counts(labels["vader_sentiment"]) if (
        labels is not None and "vader_sentiment" in labels.columns) else None

    # Agreement = share of reviews where VADER's label equals the human label,
    # i.e. VADER's accuracy on this dataset (0.7045 = 70.45%).
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
        topic_counts = [{"topic": int(k), "count": int(v), **_topic_details(k)}
                        for k, v in counts.items()]

    return {
        "total_reviews": total,
        "indexed_reviews": data.corpus_size(),
        "labelled": labelled,
        "predicted": predicted,
        "agreement": agreement,
        "aspects": aspects,
        "topics": topic_counts,
        "plates": _plate_manifest(),
    }


def _plate_manifest():
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
    """Page 02 Verdict: LIVE hybrid sentiment analysis of the typed review."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a review to analyse.")
    return sentiment.analyze(text)


@api.post("/api/summarize")
def build_summary(payload: SummaryPayload):
    """Page 06 Condense: LIVE extractive TextRank summary of the pasted review."""
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a review to summarise.")
    result = summarize.summarize(text, payload.sentences)
    result["original"] = text
    return result


@api.get("/api/aspects")
def aspect_table():
    """Page 03 Aspects: the 28-aspect ledger with polarity shares (pipeline stage 07).

    For each aspect: counts of positive / neutral / negative mentions, each as a
    share of the total, plus NET sentiment = (positive - negative) / total,
    from -1 (all negative) to +1 (all positive).
    """
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
    """Page 03 Aspects: sample review excerpts that mention one aspect."""
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
    # Random sample from the first 400 matches; the fixed random_state keeps
    # the excerpts the same between visits.
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
    """Page 04 Topics: every NMF topic with its size, keywords and polarity mix."""
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
        entry = {"topic": int(key), "count": int(count), **_topic_details(key)}
        if sentiment_column is not None:
            subset = frame.loc[frame[column] == key, sentiment_column]
            entry["sentiment"] = _polarity_counts(subset)
        topics.append(entry)
    return {"topics": topics, "total": int(len(frame))}


@api.get("/api/topics/{topic}/reviews")
def topic_reviews(topic: int, limit: int = Query(default=8, ge=1, le=30)):
    """Page 04 Topics: sample reviews whose strongest NMF topic is ``topic``."""
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
    # A fixed-seed sample from the first 500 reviews of this topic, so the same
    # reviews appear on every visit.
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
    """Page 05 Retrieval: LIVE semantic (or lexical fallback) search."""
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
    """One review of the 5,000-review corpus, by row number."""
    record = data.review_at(index)
    if record is None:
        raise HTTPException(status_code=404, detail="Review index out of range.")
    record["corpus"] = data.corpus_size()
    return record


@api.get("/api/reviews/random/pick")
def random_review():
    """A random review of the 5,000-review corpus (the "Random" buttons)."""
    size = data.corpus_size()
    if not size:
        raise HTTPException(status_code=404, detail="Corpus unavailable.")
    return review(random.randrange(size))


@api.post("/api/similar")
def similar(payload: NeighbourPayload):
    """Page 05 Retrieval, "Find similar reviews": LIVE nearest neighbours of one review."""
    result = semantic.neighbours(payload.index, payload.top_k)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    source = data.review_at(payload.index)
    if source is not None:
        source["corpus"] = data.corpus_size()  # lets the page say "Review 7 of 5,000"
    result["source"] = source
    return result


# ------------------------------------------------------------------
# Static assets
# ------------------------------------------------------------------
# Anything that is not /api/... is a file of the web interface. Unknown page
# URLs fall back to index.html so the single-page app can handle them.

@api.middleware("http")
async def always_revalidate_web_files(request, call_next):
    """Make the browser re-check the page, script and stylesheet on every load.

    Without this, a browser can keep using an old cached styles.css or app.js
    after the files are edited, and draw the new page with old code. "no-cache"
    still lets it reuse its copy when the file has not changed (via the ETag),
    so it costs almost nothing.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


@api.exception_handler(404)
def not_found(request, exc):
    if request.url.path.startswith(("/api/", "/visuals/")):
        return JSONResponse({"detail": getattr(exc, "detail", "Not found")}, status_code=404)
    return FileResponse(os.path.join(config.WEB_DIR, "index.html"))


if os.path.isdir(config.VISUALS_DIR):
    api.mount("/visuals", StaticFiles(directory=config.VISUALS_DIR), name="visuals")

# Mounted last so the /api routes above take priority over static files.
api.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
