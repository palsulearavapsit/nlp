"""Hybrid sentiment engine: VADER polarity fused with product-review rules.

The scoring logic is a faithful port of the original Streamlit dashboard, with
one small difference: the old version collapsed runs of whitespace before
matching phrases, this one matches the text as typed (so the character
positions of the evidence stay exact). A phrase split by a line break, such as
"works<newline>perfectly", therefore no longer matches. On the 100k dataset
this changes no verdicts. Three things are new:

* VADER is optional. When ``vaderSentiment`` is not installed the module falls
  back to a lexicon-only estimate and reports which engine produced the verdict,
  so the interface can be honest about fidelity instead of crashing on import.
* Every matched phrase and keyword is returned with its character span, which
  lets the frontend mark up the reviewer's own text.
* ``reasoning`` spells out, in plain English, the exact calculation that
  produced each verdict.

ROLE IN THE PROJECT
    Powers the "Verdict" page (POST /api/sentiment in app/server.py). You type a
    review; this module labels it Positive / Negative / Neutral and explains why.

NLP CONCEPT - HYBRID LEXICON-BASED SENTIMENT ANALYSIS
    Two lexicon-based scorers are combined:
      1. VADER - a general-purpose sentiment model with a ~7,500-word rated
         dictionary plus rules for negation, intensifiers ("very"),
         punctuation ("!!!"), capitals and "but". Its main output, the
         COMPOUND score, runs from -1 (most negative) to +1 (most positive).
      2. A product-review RULE LEXICON (app/lexicon.py): 122 phrases and
         45 keywords such as "waste of money" or "stopped working", given a
         fixed heavy weight. VADER scores many of these only weakly (e.g.
         "stopped working" -0.23) or not at all ("sent it back" 0.0), so
         praise elsewhere in a review can outvote them.
    No machine-learning training is involved: both are rules and word lists,
    which is why every verdict can be fully explained.

HOW WELL IT WORKS (pipeline/12_evaluate_sentiment.py, all 100k reviews)
    Agreement with the human labels: VADER alone 70.45%, this hybrid 76.27%.
    The gain is on complaints: human-negative reviews wrongly called positive
    fall from 24,180 to 17,833.

THE ALGORITHM, STEP BY STEP (analyze() below)
    1. Find every lexicon phrase and keyword in the text -> rule score
           rule = 0.40 x pos phrases + 0.08 x pos words
                - 0.50 x neg phrases - 0.10 x neg words
    2. Run VADER -> compound score (and pos / neg / neu proportions).
    3. Decide:
         only negative phrases matched -> Negative, final = min(-0.30, compound + rule)
         only positive phrases matched -> Positive, final = max(+0.30, compound + rule)
         otherwise (none, or both kinds) -> final = 0.70 x compound + 0.30 x rule
             final >= +0.05 Positive | final <= -0.05 Negative | else Neutral
    4. Hard overrides: a few phrases ("very slow", "don't buy", "waste of
       money", ...) always force Negative with a fixed score. If several
       groups match, the group listed last in HARD_NEGATIVE_OVERRIDES sets the
       score, wherever the phrases appear in the text.
    Note: under a phrase override the final score is compound + rule, which is
    NOT limited to -1..+1 (e.g. several negative phrases can give -4.2).
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

# Rule-lexicon weights (step 1). Each match adds or subtracts its weight.
POSITIVE_PHRASE_WEIGHT = 0.40
POSITIVE_WORD_WEIGHT = 0.08
NEGATIVE_PHRASE_WEIGHT = 0.50
NEGATIVE_WORD_WEIGHT = 0.10

# Blending (step 3, when no one-sided phrase override applies).
VADER_BLEND_WEIGHT = 0.70
RULE_BLEND_WEIGHT = 0.30
# Standard VADER cut-offs: scores between -0.05 and +0.05 count as neutral.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
# The weakest score a phrase override may produce, so an override never lands
# near zero.
OVERRIDE_FLOOR = 0.30

# Phrases that override the blended score outright, as in the original engine.
# (triggers, forced score). The groups are checked in this order, so if several
# match, the one listed LAST sets the score (e.g. "waste of money" -0.80 beats
# "very slow" -0.60), regardless of where each phrase appears in the text.
HARD_NEGATIVE_OVERRIDES = [
    (("very slow", "too slow", "extremely slow"), -0.60),
    (("should not be bought", "should not buy", "do not buy", "don't buy"), -0.80),
    (("waste of money", "waste money"), -0.80),
]

_WORD_RE = re.compile(r"\b[a-z]+\b")
_WHITESPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _vader():
    """Return a VADER analyzer, or ``None`` when the package is unavailable.

    lru_cache means the analyzer (and its dictionary) is built once and reused.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None
    return SentimentIntensityAnalyzer()


def engine_name():
    """Return "vader+rules" (full hybrid) or "rules-only" (VADER not installed)."""
    return "vader+rules" if _vader() is not None else "rules-only"


def normalize_text(text):
    """Lowercase and collapse runs of whitespace, so "Don't   BUY" -> "don't buy"."""
    return _WHITESPACE_RE.sub(" ", str(text).lower()).strip()


def _spans(haystack_lower, needle, whole_word):
    """Character spans of every occurrence of ``needle`` in the lowered text."""
    # \b = word boundary: with whole_word, "bad" does not match inside "badge".
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


def lexicon_rule_score(text):
    """Step 1: rule-lexicon score plus the evidence that produced it.

    Returns (score, negative_phrases, positive_phrases, negative_words,
    positive_words). Each evidence list holds {"term", "spans"} dicts. Each
    distinct entry counts once, however many times it appears.
    """
    lowered = str(text).lower()

    # Phrases are matched as substrings (see app/lexicon.py for the caveat).
    negative_phrases = _match(lowered, NEGATIVE_PHRASES, whole_word=False)
    positive_phrases = _match(lowered, POSITIVE_PHRASES, whole_word=False)

    # Keywords must be whole words: first tokenize into a set of words, keep
    # the lexicon words present, then locate them for highlighting.
    words = set(_WORD_RE.findall(lowered))
    negative_words = _match(lowered, [w for w in NEGATIVE_WORDS if w in words], True)
    positive_words = _match(lowered, [w for w in POSITIVE_WORDS if w in words], True)

    score = (
        len(positive_phrases) * POSITIVE_PHRASE_WEIGHT
        + len(positive_words) * POSITIVE_WORD_WEIGHT
        - len(negative_phrases) * NEGATIVE_PHRASE_WEIGHT
        - len(negative_words) * NEGATIVE_WORD_WEIGHT
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


def _plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def analyze(text):
    """Classify a review and return the full evidence trail behind the verdict."""
    text = str(text)
    reasoning = []  # plain-English account of each step, shown in the UI

    # ---- Step 1: product-review rule lexicon ------------------------------
    (
        rule_score,
        negative_phrases,
        positive_phrases,
        negative_words,
        positive_words,
    ) = lexicon_rule_score(text)
    reasoning.append(
        f"Rule lexicon: {_plural(len(positive_phrases), 'positive phrase')} × 0.40 "
        f"+ {_plural(len(positive_words), 'positive keyword')} × 0.08 "
        f"- {_plural(len(negative_phrases), 'negative phrase')} × 0.50 "
        f"- {_plural(len(negative_words), 'negative keyword')} × 0.10 "
        f"= rule score {rule_score:+.3f}."
    )

    # ---- Step 2: VADER (or the lexicon-only stand-in) ---------------------
    analyzer = _vader()
    if analyzer is not None:
        # {"neg": .., "neu": .., "pos": .., "compound": ..}; pos + neu + neg = 1
        scores = analyzer.polarity_scores(text)
    else:
        scores = _lexicon_polarity(
            rule_score,
            len(negative_phrases) + len(negative_words),
            len(positive_phrases) + len(positive_words),
        )
    compound = scores["compound"]
    if analyzer is not None:
        reasoning.append(
            f"VADER compound score = {compound:+.3f} "
            f"(-1 = most negative, +1 = most positive)."
        )
    else:
        reasoning.append(
            f"VADER is not installed, so the rule score clipped to -1..+1 stands in "
            f"for its compound score: {compound:+.3f}."
        )

    # ---- Step 3: combine --------------------------------------------------
    # Explicit product-review phrasing outranks the general-purpose model.
    if negative_phrases and not positive_phrases:
        sentiment = NEGATIVE
        final_score = min(-OVERRIDE_FLOOR, compound + rule_score)
        basis = "negative phrase override"
        reasoning.append(
            f"Only negative phrases matched, so the negative phrase override applies: "
            f"final = min(-0.30, compound + rule score) "
            f"= min(-0.30, {compound + rule_score:+.3f}) = {final_score:+.3f} → Negative."
        )
    elif positive_phrases and not negative_phrases:
        sentiment = POSITIVE
        final_score = max(OVERRIDE_FLOOR, compound + rule_score)
        basis = "positive phrase override"
        reasoning.append(
            f"Only positive phrases matched, so the positive phrase override applies: "
            f"final = max(+0.30, compound + rule score) "
            f"= max(+0.30, {compound + rule_score:+.3f}) = {final_score:+.3f} → Positive."
        )
    else:
        # No phrase matched, or phrases of BOTH polarities did: blend the two
        # scores, trusting VADER more (70%) than the rule lexicon (30%).
        final_score = compound * VADER_BLEND_WEIGHT + rule_score * RULE_BLEND_WEIGHT
        if final_score >= POSITIVE_THRESHOLD:
            sentiment = POSITIVE
        elif final_score <= NEGATIVE_THRESHOLD:
            sentiment = NEGATIVE
        else:
            sentiment = NEUTRAL
        basis = "blended polarity"
        situation = ("Phrases of both polarities matched"
                     if negative_phrases and positive_phrases
                     else "No lexicon phrase matched")
        reasoning.append(
            f"{situation}, so the two scores are blended: "
            f"final = 0.70 × compound + 0.30 × rule score "
            f"= 0.70 × {compound:+.3f} + 0.30 × {rule_score:+.3f} = {final_score:+.3f}. "
            f"At or above +0.05 means Positive, at or below -0.05 Negative, "
            f"in between Neutral → {sentiment}."
        )

    # ---- Step 4: hard overrides -------------------------------------------
    # A handful of phrases are decisive in a product review whatever else the
    # text says ("amazing design but very slow" is still a complaint).
    lowered = normalize_text(text)
    for triggers, forced_score in HARD_NEGATIVE_OVERRIDES:
        matched_trigger = next((t for t in triggers if t in lowered), None)
        if matched_trigger is not None:
            sentiment = NEGATIVE
            final_score = forced_score
            basis = "hard rule override"
            reasoning.append(
                f"Hard rule: the text contains “{matched_trigger}”, so the result "
                f"above is replaced: Negative with a fixed score of {forced_score:+.2f}."
            )

    return {
        "text": text,
        "sentiment": sentiment,          # the verdict: Positive / Negative / Neutral
        "final_score": round(float(final_score), 4),
        "compound": round(float(compound), 4),        # VADER overall score, -1..+1
        "positive": round(float(scores["pos"]), 4),   # VADER: share of positive text
        "negative": round(float(scores["neg"]), 4),   # VADER: share of negative text
        "neutral": round(float(scores["neu"]), 4),    # VADER: share of neutral text
        "rule_score": round(float(rule_score), 4),    # step 1 result
        "basis": basis,                  # which rule in step 3/4 decided the verdict
        "engine": engine_name(),
        "reasoning": reasoning,
        "evidence": {                    # what to highlight in the review text
            "negative_phrases": negative_phrases,
            "positive_phrases": positive_phrases,
            "negative_words": negative_words,
            "positive_words": positive_words,
        },
    }
