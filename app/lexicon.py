"""Product-review sentiment lexicons.

Extracted verbatim from the original Streamlit dashboard, so the hybrid
VADER + rule engine uses exactly the same lists (see app/sentiment.py for the
one small matching difference).

WHAT A "LEXICON" IS
    In NLP a lexicon is a hand-made list of words or phrases with a known
    meaning - here, a known sentiment. Lexicon-based sentiment analysis means
    finding those entries in a text and adding up their effect.

WHY THE PROJECT NEEDS ITS OWN LEXICON ON TOP OF VADER
    VADER's dictionary was built for general text (social media). Many
    shopper complaints score only weakly negative or neutral in VADER:
    "stopped working" scores -0.23, "sent it back" and "returned this" score
    0.0. In a longer review, praise words then outvote them: "The screen is
    great and the design is lovely but it stopped working after a week."
    scores +0.38 in VADER, i.e. positive. In pipeline stage 06, VADER labelled
    24,180 of the 50,000 human-negative reviews as positive. These lists give
    product-review phrases a fixed, heavy weight so they are not outvoted.
    (Many single entries such as "excellent" or "worst" are also in VADER's
    own dictionary; here they get this extra weight on top.)

HOW EACH LIST IS USED (see app/sentiment.py -> lexicon_rule_score())
    list               entries  weight per match  matched as
    POSITIVE_PHRASES      48        +0.40         substring
    NEGATIVE_PHRASES      74        -0.50         substring
    POSITIVE_WORDS        19        +0.08         whole word
    NEGATIVE_WORDS        26        -0.10         whole word
    Phrases weigh far more than single words: a phrase is unambiguous
    ("works perfectly"), a single word is not ("good" inside "not good").
    Negative weights are larger than positive ones, which suits this data
    because complaints are what VADER tends to miss.

    "Substring" means a phrase also matches inside a longer word or phrase, and
    overlapping entries each count: "worst product" matches both "worst" and
    "worst product", and "broken" matches both "broken" and "broke".
"""

NEGATIVE_PHRASES = [
    "waste of money",
    "waste money",
    "waste of time",
    "waste time",
    "not worth",
    "not worth the money",
    "do not buy",
    "do not purchase",
    "don't buy",
    "dont buy",
    "don't purchase",
    "dont purchase",
    "should not buy",
    "should not be bought",
    "shouldn't buy",
    "shouldn't be bought",
    "avoid this",
    "avoid this product",
    "stay away",
    "never buy",
    "never purchase",
    "poor quality",
    "terrible quality",
    "bad quality",
    "low quality",
    "very slow",
    "too slow",
    "extremely slow",
    "does not work",
    "doesn't work",
    "doesnt work",
    "not working",
    "stopped working",
    "barely works",
    "hardly works",
    "broken",
    "broke",
    "defective",
    "useless",
    "disappointed",
    "disappointing",
    "disappointment",
    "horrible",
    "terrible",
    "awful",
    "worst",
    "worst product",
    "worst purchase",
    "bad product",
    "bad purchase",
    "poor product",
    "poor performance",
    "poor service",
    "poor design",
    "poor sound",
    "poor battery",
    "poor screen",
    "poor material",
    "too expensive",
    "overpriced",
    "not recommended",
    "would not recommend",
    "wouldn't recommend",
    "do not recommend",
    "not good",
    "not great",
    "not happy",
    "not satisfied",
    "very disappointing",
    "highly disappointing",
    "return this",
    "returned this",
    "sent it back",
    "send it back",
]

POSITIVE_PHRASES = [
    "highly recommend",
    "strongly recommend",
    "would recommend",
    "definitely recommend",
    "must buy",
    "great product",
    "excellent product",
    "excellent quality",
    "great quality",
    "good quality",
    "amazing quality",
    "fantastic quality",
    "works perfectly",
    "works great",
    "works well",
    "works perfectly well",
    "very good",
    "very great",
    "really good",
    "really great",
    "extremely good",
    "extremely great",
    "very happy",
    "very satisfied",
    "really happy",
    "really satisfied",
    "love this",
    "love the product",
    "love it",
    "excellent",
    "amazing",
    "fantastic",
    "awesome",
    "wonderful",
    "perfect",
    "best product",
    "best purchase",
    "worth the money",
    "worth every penny",
    "good value",
    "great value",
    "excellent value",
    "fast delivery",
    "quick delivery",
    "easy to use",
    "easy to install",
    "easy to setup",
    "high quality",
]

NEGATIVE_WORDS = [
    "bad",
    "terrible",
    "horrible",
    "awful",
    "worst",
    "poor",
    "broken",
    "useless",
    "defective",
    "disappointed",
    "disappointing",
    "slow",
    "expensive",
    "overpriced",
    "hate",
    "refund",
    "return",
    "problem",
    "problems",
    "failure",
    "fail",
    "failed",
    "damage",
    "damaged",
    "annoying",
    "waste",
]

POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "fantastic",
    "wonderful",
    "perfect",
    "love",
    "best",
    "happy",
    "satisfied",
    "recommend",
    "recommended",
    "easy",
    "quality",
    "fast",
    "useful",
    "value",
]

