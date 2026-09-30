"""Tests for the three live NLP engines (no web server needed).

They double as executable documentation: each test states one rule that the
explanations in the code and in PROJECT_GUIDE.md rely on.

    pip install pytest
    python -m pytest
"""

import numpy as np
import pytest

from app import semantic, sentiment, summarize


# ---------------------------------------------------------------------------
# Sentiment (app/sentiment.py)
# ---------------------------------------------------------------------------

def test_hard_rule_forces_negative_even_with_praise():
    result = sentiment.analyze("Amazing design, but very slow.")
    assert result["sentiment"] == "Negative"
    assert result["basis"] == "hard rule override"
    assert result["final_score"] == -0.60


def test_last_matching_hard_rule_sets_the_score():
    # "very slow" forces -0.60, then "waste of money" (a later rule) forces -0.80.
    result = sentiment.analyze("Very slow and a waste of money.")
    assert result["final_score"] == -0.80


def test_only_positive_phrases_trigger_positive_override():
    result = sentiment.analyze("Works perfectly and I would highly recommend it.")
    assert result["sentiment"] == "Positive"
    assert result["basis"] == "positive phrase override"
    assert result["final_score"] >= 0.30


def test_text_without_opinion_is_neutral():
    result = sentiment.analyze("It arrived on Tuesday. The box contained two cables.")
    assert result["sentiment"] == "Neutral"
    assert result["basis"] == "blended polarity"


def test_blended_score_is_70_percent_vader_30_percent_rules():
    # No lexicon PHRASE here, only keywords, so the blend formula applies.
    result = sentiment.analyze("I hate how good this is.")
    assert result["basis"] == "blended polarity"
    expected = 0.70 * result["compound"] + 0.30 * result["rule_score"]
    assert result["final_score"] == pytest.approx(expected, abs=1e-3)


def test_rule_score_uses_the_documented_weights():
    # 1 negative phrase ("not worth", which also contains no keyword) = -0.50
    score, neg_phrases, pos_phrases, neg_words, pos_words = \
        sentiment.lexicon_rule_score("not worth it")
    assert [p["term"] for p in neg_phrases] == ["not worth"]
    assert not (pos_phrases or neg_words or pos_words)
    assert score == pytest.approx(-0.50)


def test_evidence_spans_point_at_the_matched_text():
    text = "Honestly a waste of money."
    result = sentiment.analyze(text)
    phrase = next(p for p in result["evidence"]["negative_phrases"]
                  if p["term"] == "waste of money")
    start, end = phrase["spans"][0]
    assert text.lower()[start:end] == "waste of money"


def test_reasoning_explains_every_verdict():
    result = sentiment.analyze("The massager is very slow. Complete waste of money.")
    steps = result["reasoning"]
    assert steps[0].startswith("Rule lexicon")
    assert any("VADER" in step for step in steps)
    assert "waste of money" in steps[-1]  # the deciding step comes last


# ---------------------------------------------------------------------------
# Summarization (app/summarize.py)
# ---------------------------------------------------------------------------

LONG_REVIEW = (
    "I bought this blender last month after my old one died. The motor is powerful and it "
    "crushes ice without any trouble. However, the lid leaks when the jar is more than half full. "
    "Cleaning is easy because the blades come out. The price was reasonable for what you get. "
    "Customer service replied quickly when I asked about the lid. Overall I am satisfied, but "
    "I wish the design had been tested more carefully before release. I would buy it again."
)


def test_short_reviews_are_returned_unchanged():
    result = summarize.summarize("Great blender. Works well.", 2)
    assert result["summary"] == "Great blender. Works well."
    assert result["engine"] == "verbatim (already concise)"


def test_summary_is_extractive():
    # Every summary sentence is copied from the original text.
    result = summarize.summarize(LONG_REVIEW, 2)
    for sentence in summarize.split_sentences(result["summary"]):
        assert sentence in LONG_REVIEW
    assert 0 < result["compression"] < 1


def test_builtin_textrank_returns_top_sentences_in_reading_order():
    sentences = summarize.split_sentences(LONG_REVIEW)
    picked = summarize._textrank(sentences, 3)
    assert len(picked) == 3
    assert picked == sorted(picked)


# ---------------------------------------------------------------------------
# Semantic similarity (app/semantic.py)
# ---------------------------------------------------------------------------

def test_cosine_similarity_measures_direction_not_length():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 0.0]], dtype="float32")
    scores = semantic.cosine_similarity([2.0, 0.0], matrix)
    assert scores == pytest.approx([1.0, 0.0, 1.0], abs=1e-6)


def test_neighbour_ranking_skips_the_query_review_itself():
    scores = np.array([0.2, 1.0, 0.9, 0.5])
    assert semantic._top_k_indices(scores, 2, exclude=1) == [2, 3]


def test_semantic_search_ranks_the_closest_embedding_first(monkeypatch):
    # Stand-in for MiniLM (so the test runs without sentence-transformers):
    # it "encodes" every query as the stored embedding of review 123, so a
    # correct semantic search must return review 123 first, with similarity 1.
    from app import data

    matrix = data.embeddings()
    if matrix is None:
        pytest.skip("review_embeddings.csv not available")

    class StubEncoder:
        def encode(self, query, convert_to_numpy=True):
            return matrix[123]

    monkeypatch.setattr(semantic, "_sentence_model", lambda: StubEncoder())
    result = semantic.search("any query", top_k=3)
    assert result["mode"] == "semantic"
    assert result["results"][0]["index"] == 123
    assert result["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)
