"""Extractive summarisation via TextRank.

Uses ``sumy``'s TextRank when installed. Otherwise it runs a self-contained
TextRank over TF-based sentence similarity, so the module never hard-fails on a
machine without the optional dependency.
"""

import math
import re
from collections import Counter

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_TOKEN_RE = re.compile(r"[a-z']+")

MIN_WORDS_TO_SUMMARIZE = 25

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he her here hers herself him himself his how i i'd i'll i'm i've if in
into is isn't it it's its itself let's me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she should
shouldn't so some such than that the their theirs them themselves then there these they
this those through to too under until up very was wasn't we were weren't what when where
which while who whom why with won't would wouldn't you your yours yourself yourselves
""".split())


def split_sentences(text):
    parts = [part.strip() for part in _SENTENCE_RE.split(str(text).strip())]
    return [part for part in parts if part]


def _tokenize(sentence):
    return [t for t in _TOKEN_RE.findall(sentence.lower()) if t not in STOPWORDS]


def _similarity(a, b):
    """Cosine-style overlap between two sentence token bags."""
    if not a or not b:
        return 0.0
    shared = sum(min(a[t], b[t]) for t in a.keys() & b.keys())
    if not shared:
        return 0.0
    return shared / (math.sqrt(sum(a.values())) * math.sqrt(sum(b.values())))


def _textrank(sentences, sentence_count, damping=0.85, iterations=40):
    """Rank sentences by PageRank over a similarity graph."""
    bags = [Counter(_tokenize(s)) for s in sentences]
    size = len(sentences)

    weights = [[_similarity(bags[i], bags[j]) if i != j else 0.0
                for j in range(size)] for i in range(size)]
    totals = [sum(row) or 1.0 for row in weights]

    scores = [1.0 / size] * size
    for _ in range(iterations):
        scores = [
            (1 - damping) / size
            + damping * sum(weights[j][i] / totals[j] * scores[j] for j in range(size))
            for i in range(size)
        ]

    ranked = sorted(range(size), key=lambda i: -scores[i])[:sentence_count]
    return sorted(ranked)


def _sumy_summary(text, sentence_count):
    """TextRank via sumy, or ``None`` if sumy is unavailable or fails."""
    try:
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.summarizers.text_rank import TextRankSummarizer
    except ImportError:
        return None

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        picked = TextRankSummarizer()(parser.document, sentences_count=sentence_count)
        joined = " ".join(str(sentence) for sentence in picked)
        return joined or None
    except Exception:
        return None


def summarize(text, sentence_count=3):
    """Return the summary plus the compression statistics behind it."""
    text = str(text).strip()
    original_words = len(text.split())
    sentences = split_sentences(text)

    if original_words < MIN_WORDS_TO_SUMMARIZE or len(sentences) <= sentence_count:
        return _result(text, text, sentences, len(sentences), "verbatim (already concise)")

    summary = _sumy_summary(text, sentence_count)
    engine = "sumy textrank"

    if summary is None:
        picked = _textrank(sentences, sentence_count)
        summary = " ".join(sentences[i] for i in picked)
        engine = "textrank (built-in)"

    return _result(text, summary, sentences, sentence_count, engine)


def _result(text, summary, sentences, kept, engine):
    original_words = len(text.split())
    summary_words = len(summary.split())
    ratio = summary_words / original_words if original_words else 1.0
    return {
        "summary": summary,
        "engine": engine,
        "original_words": original_words,
        "summary_words": summary_words,
        "original_sentences": len(sentences),
        "summary_sentences": kept,
        "compression": round(1 - ratio, 3),
    }
