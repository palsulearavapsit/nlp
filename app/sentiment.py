"""Hybrid sentiment engine: VADER polarity fused with product-review rules.

The scoring logic is a faithful port of the original Streamlit dashboard. Two
things are new:

* VADER is optional. When ``vaderSentiment`` is not installed the module falls
  back to a lexicon-only estimate and reports which engine produced the verdict,
  so the interface can be honest about fidelity instead of crashing on import.
* Every matched phrase and keyword is returned with its character span, which
  lets the frontend mark up the reviewer's own text.
"""

import re
from functools import lru_cache

from app.lexicon import (
    NEGATIVE_PHRASES,
    NEGATIVE_WORDS,
    POSITIVE_PHRASES,
    POSITIVE_WORDS,
)

POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"

# Phrases that override the blended score outright, as in the original engine.
HARD_NEGATIVE_OVERRIDES = [
    (("very slow", "too slow", "extremely slow"), -0.60),
    (("should not be bought", "should not buy", "do not buy", "don't buy"), -0.80),
    (("waste of money", "waste money"), -0.80),
]

_WORD_RE = re.compile(r"\b[a-z]+\b")
_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _vader():
    """Return a VADER analyzer, or ``None`` when the package is unavailable."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    return SentimentIntensityAnalyzer()


def engine_name():
    return "vader+rules" if _vader() is not None else "rules-only"


def normalize_text(text):
    return _WHITESPACE_RE.sub(" ", str(text).lower()).strip()


def _spans(haystack_lower, needle, whole_word):
    """Character spans of every occurrence of ``needle`` in the lowered text."""
    pattern = rf"\b{re.escape(needle)}\b" if whole_word else re.escape(needle)
    return [[m.start(), m.end()] for m in re.finditer(pattern, haystack_lower)]


def _match(text_lower, terms, whole_word):
    """Matched terms with their spans, longest term first so markup nests cleanly."""
    found = []
    for term in terms:
        spans = _spans(text_lower, term, whole_word)
        if spans:
            found.append({"term": term, "spans": spans})
    found.sort(key=lambda item: -len(item["term"]))
    return found


def phrase_score(text):
    """Rule-engine score plus the evidence that produced it."""
    lowered = str(text).lower()

    negative_phrases = _match(lowered, NEGATIVE_PHRASES, whole_word=False)
    positive_phrases = _match(lowered, POSITIVE_PHRASES, whole_word=False)

    words = set(_WORD_RE.findall(lowered))
    negative_words = _match(lowered, [w for w in NEGATIVE_WORDS if w in words], True)
    positive_words = _match(lowered, [w for w in POSITIVE_WORDS if w in words], True)

    score = (
        len(positive_phrases) * 0.40
        + len(positive_words) * 0.08
        - len(negative_phrases) * 0.50
        - len(negative_words) * 0.10
    )
    return score, negative_phrases, positive_phrases, negative_words, positive_words


def _lexicon_polarity(rule_score, negative_hits, positive_hits):
    """Stand-in for VADER's pos/neg/neu/compound when the package is missing."""
    total = negative_hits + positive_hits
    if total == 0:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

    pos = positive_hits / total
    neg = negative_hits / total
    intensity = min(1.0, total / 6.0)
    return {
        "pos": round(pos * intensity, 3),
        "neg": round(neg * intensity, 3),
        "neu": round(1.0 - intensity, 3),
        "compound": max(-1.0, min(1.0, rule_score)),
    }


def analyze(text):
    """Classify a review and return the full evidence trail behind the verdict."""
    text = str(text)
    (
        rule_score,
        negative_phrases,
        positive_phrases,
        negative_words,
        positive_words,
    ) = phrase_score(text)

    analyzer = _vader()
    if analyzer is not None:
        scores = analyzer.polarity_scores(text)
    else:
        scores = _lexicon_polarity(
            rule_score,
            len(negative_phrases) + len(negative_words),
            len(positive_phrases) + len(positive_words),
        )
    compound = scores["compound"]

    # Explicit product-review phrasing outranks the general-purpose model.
    if negative_phrases and not positive_phrases:
        sentiment = NEGATIVE
        final_score = min(-0.30, compound + rule_score)
        basis = "negative phrase override"
    elif positive_phrases and not negative_phrases:
        sentiment = POSITIVE
        final_score = max(0.30, compound + rule_score)
        basis = "positive phrase override"
    else:
        final_score = compound * 0.70 + rule_score * 0.30
        if final_score >= 0.05:
            sentiment = POSITIVE
        elif final_score <= -0.05:
            sentiment = NEGATIVE
        else:
            sentiment = NEUTRAL
        basis = "blended polarity"

    lowered = normalize_text(text)
    for triggers, forced_score in HARD_NEGATIVE_OVERRIDES:
        if any(trigger in lowered for trigger in triggers):
            sentiment = NEGATIVE
            final_score = forced_score
            basis = "hard rule override"

    return {
        "text": text,
        "sentiment": sentiment,
        "final_score": round(float(final_score), 4),
        "compound": round(float(compound), 4),
        "positive": round(float(scores["pos"]), 4),
        "negative": round(float(scores["neg"]), 4),
        "neutral": round(float(scores["neu"]), 4),
        "rule_score": round(float(rule_score), 4),
        "basis": basis,
        "engine": engine_name(),
        "evidence": {
            "negative_phrases": negative_phrases,
            "positive_phrases": positive_phrases,
            "negative_words": negative_words,
            "positive_words": positive_words,
        },
    }
