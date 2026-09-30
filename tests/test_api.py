"""Smoke tests for the HTTP API (app/server.py), using the real CSV files.

    python -m pytest
"""

import pytest

pytest.importorskip("httpx")  # FastAPI's TestClient needs httpx

from fastapi.testclient import TestClient  # noqa: E402

from app.server import api  # noqa: E402

client = TestClient(api)


def test_sentiment_endpoint_returns_verdict_evidence_and_reasoning():
    response = client.post("/api/sentiment", json={"text": "Do not buy. It broke in a week."})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "Negative"
    assert body["evidence"]["negative_phrases"]
    assert body["reasoning"]


def test_empty_text_is_rejected():
    assert client.post("/api/sentiment", json={"text": ""}).status_code == 422


def test_topics_carry_nmf_keywords_and_labels():
    topics = client.get("/api/topics").json()["topics"]
    assert len(topics) == 10
    video_games = next(t for t in topics if t["topic"] == 10)
    assert video_games["label"] == "Video games"
    assert video_games["keywords"][0] == "game"


def test_overview_reports_the_corpus_and_vader_agreement():
    body = client.get("/api/overview").json()
    assert body["total_reviews"] == 100000
    assert body["labelled"] == {"positive": 50000, "negative": 50000, "neutral": 0}
    assert body["agreement"] == pytest.approx(0.7045, abs=1e-4)


def test_neighbours_never_include_the_source_review():
    body = client.post("/api/similar", json={"index": 7, "top_k": 5}).json()
    assert body["mode"] == "semantic"
    assert 7 not in [result["index"] for result in body["results"]]
    assert body["source"]["corpus"] == 5000  # page shows "Review 7 of 5,000"
