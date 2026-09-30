"""Extractive summarisation via TextRank.

Uses ``sumy``'s TextRank when installed. Otherwise it runs a self-contained
TextRank over TF-based sentence similarity, so the module never hard-fails on a
machine without the optional dependency.

ROLE IN THE PROJECT
    Powers the "Condense" page (POST /api/summarize in app/server.py). You
    paste a long review; you get back its most important sentences plus
    statistics on how much shorter the summary is.

NLP CONCEPT - EXTRACTIVE SUMMARIZATION
    The summary is made of sentences COPIED from the original, never new text.
    It cannot invent anything; it only decides which sentences matter most.

NLP CONCEPT - TEXTRANK (Mihalcea & Tarau, 2004)
    TextRank is Google's PageRank applied to sentences instead of web pages:
      1. Split the review into sentences.
      2. Turn each sentence into a list of words.
      3. Build a graph: every sentence is a node; the edge between two
         sentences is weighted by how many words they share.
      4. Run PageRank on the graph (damping factor 0.85). A sentence scores
         high when it is similar to many other high-scoring sentences - it is
         "central" to the review.
      5. Keep the top-N sentences and print them in their ORIGINAL order so
         the summary still reads naturally.

TWO IMPLEMENTATIONS OF THE SAME IDEA
    live engine (sumy's TextRankSummarizer, used whenever sumy is installed):
        keeps stopwords; edge weight = shared words / (log|A| + log|B|);
        repeats PageRank until the scores stop changing (converge).
    built-in fallback (the functions below, used only without sumy):
        drops stopwords (_tokenize); edge weight = shared words /
        (sqrt|A| x sqrt|B|) (_sentence_similarity); runs PageRank for a fixed
        40 rounds (_textrank). Written out in full so the maths is visible.
    |A| = number of words in sentence A.
"""

import math
import re
from collections import Counter

# Sentence boundary: a space that follows . ! or ? AND precedes a capital
# letter, quote or bracket. "(?<=...)" and "(?=...)" check the neighbours
# without consuming them, so the punctuation stays with its sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_TOKEN_RE = re.compile(r"[a-z']+")

# Reviews shorter than this are returned unchanged - nothing to condense.
MIN_WORDS_TO_SUMMARIZE = 25

# Stopwords: very frequent function words that would make every pair of
# sentences look similar ("the", "and", "it"), so they are ignored.
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
    """Step 1 - sentence segmentation: split text into a list of sentences."""
    parts = [part.strip() for part in _SENTENCE_RE.split(str(text).strip())]
    return [part for part in parts if part]


def _tokenize(sentence):
    """Step 2 - lowercase word tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(sentence.lower()) if t not in STOPWORDS]


def _sentence_similarity(a, b):
    """Step 3 - edge weight between two sentences (``a``, ``b`` are word-count bags).

        similarity = shared words / (sqrt(words in a) x sqrt(words in b))

    Dividing by the sentence sizes stops long sentences from winning just
    because they contain more words.
    """
    if not a or not b:
        return 0.0
    shared = sum(min(a[t], b[t]) for t in a.keys() & b.keys())
    if not shared:
        return 0.0
    return shared / (math.sqrt(sum(a.values())) * math.sqrt(sum(b.values())))


def _textrank(sentences, sentence_count, damping=0.85, iterations=40):
    """Step 4 - rank sentences by PageRank over a similarity graph.

    PageRank update, repeated ``iterations`` times for every sentence i:

        score(i) = (1 - d) / n  +  d x SUM over j of [ w(j, i) / total_w(j) x score(j) ]

    d = 0.85 is the damping factor (the standard PageRank value), n the number
    of sentences, w(j, i) the similarity edge, and total_w(j) the sum of j's
    edges - so each sentence shares its score out among its neighbours in
    proportion to how similar they are. 40 rounds is plenty for the scores to
    settle (converge) on review-sized graphs.

    Returns the indices of the top ``sentence_count`` sentences, in text order.
    """
    bags = [Counter(_tokenize(s)) for s in sentences]
    size = len(sentences)

    # weights[i][j] = similarity between sentence i and j (0 on the diagonal:
    # a sentence does not vote for itself).
    weights = [[_sentence_similarity(bags[i], bags[j]) if i != j else 0.0
                for j in range(size)] for i in range(size)]
    totals = [sum(row) or 1.0 for row in weights]

    scores = [1.0 / size] * size  # start with every sentence equally important
    for _ in range(iterations):
        scores = [
            (1 - damping) / size
            + damping * sum(weights[j][i] / totals[j] * scores[j] for j in range(size))
            for i in range(size)
        ]

    # Step 5 - best sentences first, keep the top N, then restore reading order.
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

    # Too short to condense, or already no longer than the requested summary.
    if original_words < MIN_WORDS_TO_SUMMARIZE or len(sentences) <= sentence_count:
        return _result(text, text, sentences, len(sentences), "verbatim (already concise)")

    # Preferred engine: the sumy library's TextRank (same library as pipeline stage 09).
    summary = _sumy_summary(text, sentence_count)
    engine = "sumy textrank"

    # Fallback: the built-in TextRank above.
    if summary is None:
        picked = _textrank(sentences, sentence_count)
        summary = " ".join(sentences[i] for i in picked)
        engine = "textrank (built-in)"

    return _result(text, summary, sentences, sentence_count, engine)


def _result(text, summary, sentences, kept, engine):
    """Package the summary with its statistics.

    compression = 1 - summary words / original words, i.e. the share of the
    text that was removed (0.65 means the summary is 65% shorter).
    """
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
