"""Filesystem locations and tunable limits for the NLP service.

Every other module imports its paths from here, so this file is the map of
which data file feeds which feature. The CSVs themselves were produced by the
offline scripts in pipeline/ (the script that wrote each one is noted below).
"""

import os

# Project root = the folder that contains app/, web/ and the CSV files.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEB_DIR = os.path.join(BASE_DIR, "web")          # HTML / CSS / JS served to the browser
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")  # chart images shown on the Corpus page
CACHE_DIR = os.path.join(BASE_DIR, ".cache")     # fast binary copy of the embeddings

# ---------------------------------------------------------------------------
# Data files, and the pipeline stage that produced each one
# ---------------------------------------------------------------------------
# 100k reviews + human label + VADER label/score        (pipeline/06_sentiment_analysis.py)
SENTIMENT_FILE = os.path.join(BASE_DIR, "amazon_reviews_sentiment.csv")
# one row per (aspect, review) pair                      (pipeline/07_aspect_sentiment.py)
ASPECT_FILE = os.path.join(BASE_DIR, "amazon_review_aspects.csv")
# 28 aspects x negative/neutral/positive counts          (pipeline/07_aspect_sentiment.py)
ASPECT_SUMMARY_FILE = os.path.join(BASE_DIR, "aspect_sentiment_summary.csv")
# 100k reviews + NMF topic number 1..10                  (pipeline/08_topic_extraction.py)
TOPIC_FILE = os.path.join(BASE_DIR, "amazon_reviews_topics.csv")
# top 10 NMF terms for each topic                        (pipeline/08_topic_extraction.py)
TOPIC_KEYWORDS_FILE = os.path.join(BASE_DIR, "topic_keywords.csv")
# the 5,000-review retrieval corpus + TextRank summaries (pipeline/09_summarization.py)
SUMMARY_FILE = os.path.join(BASE_DIR, "amazon_reviews_summarized.csv")
# 5,000 x 384 sentence embeddings, row i = review i of SUMMARY_FILE
#                                                        (pipeline/10_embeddings_similarity.py)
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "review_embeddings.csv")
# The same matrix saved in NumPy's binary format on first load (much faster to read).
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "review_embeddings.npy")

# Sentence-transformer checkpoint used to build review_embeddings.csv.
# A search query MUST be encoded with this same model, otherwise query and
# review vectors live in different spaces and cannot be compared.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

MAX_INPUT_CHARS = 20000   # longest text the sentiment / summary endpoints accept
MAX_RESULTS = 25          # most results a search or neighbour lookup may return
DEFAULT_RESULTS = 5

# NMF (pipeline/08) only outputs NUMBERED topics. These short names were written
# by people after reading each topic's top keywords in topic_keywords.csv. They
# are an interpretation, which is standard practice in topic modelling.
TOPIC_LABELS = {
    1: "Novels & stories",
    2: "Books & authors",
    3: "Movies",
    4: "Products & money",
    5: "“Great” praise",
    6: "Music CDs & singers",
    7: "DVD & Blu-ray editions",
    8: "Albums & bands",
    9: "“Good” / lukewarm",
    10: "Video games",
}

# "Plates" are the six chart images at the bottom of the Corpus page, drawn by
# app/plates.py.
# Each entry: (id, title, image file in visuals/, caption shown under the chart).
PLATES = [
    ("sentiment_distribution", "Sentiment Distribution", "sentiment_distribution.png",
     "How many of the 100,000 reviews VADER labelled positive, neutral or negative "
     "(pipeline stage 06). The human labels are exactly 50/50, so the tall positive "
     "bar shows VADER's positive bias."),
    ("topic_distribution", "Topic Distribution", "topic_distribution.png",
     "How many reviews NMF topic modelling assigned to each of the 10 topics "
     "(stage 08). Each review counts once, under its strongest topic."),
    ("aspect_frequency", "Aspect Frequency", "aspect_frequency.png",
     "How many reviews mention each of the 28 aspect words (stage 07). "
     "One review can mention several aspects."),
    ("aspect_sentiment", "Aspect Sentiment", "aspect_sentiment.png",
     "For the 16 most-mentioned aspects: what share of the reviews mentioning "
     "each one are positive, neutral or negative (stage 07)."),
    ("review_length_distribution", "Review Length", "review_length_distribution.png",
     "How long reviews are, in words. The red line marks the median; the longest "
     "1% are trimmed so the shape stays readable."),
    ("top_words", "Top Words", "top_words.png",
     "The most frequent words in the first 40,000 reviews once stopwords are "
     "removed. Books and movies dominate: book, movie, read and story are near the top."),
]
