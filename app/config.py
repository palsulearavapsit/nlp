"""Filesystem locations and tunable limits for the NLP service."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEB_DIR = os.path.join(BASE_DIR, "web")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

SENTIMENT_FILE = os.path.join(BASE_DIR, "amazon_reviews_sentiment.csv")
ASPECT_FILE = os.path.join(BASE_DIR, "amazon_review_aspects.csv")
ASPECT_SUMMARY_FILE = os.path.join(BASE_DIR, "aspect_sentiment_summary.csv")
TOPIC_FILE = os.path.join(BASE_DIR, "amazon_reviews_topics.csv")
SUMMARY_FILE = os.path.join(BASE_DIR, "amazon_reviews_summarized.csv")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "review_embeddings.csv")
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "review_embeddings.npy")

# Sentence-transformer checkpoint used to build review_embeddings.csv.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

MAX_INPUT_CHARS = 20000
MAX_RESULTS = 25
DEFAULT_RESULTS = 5

# Plates rendered by the analytics notebook, surfaced in the Atlas module.
PLATES = [
    ("sentiment_distribution", "Sentiment Distribution", "sentiment_distribution.png",
     "Share of reviews by predicted polarity across the full corpus."),
    ("topic_distribution", "Topic Distribution", "topic_distribution.png",
     "Volume of reviews assigned to each latent topic cluster."),
    ("aspect_frequency", "Aspect Frequency", "aspect_frequency.png",
     "How often each product aspect is named by reviewers."),
    ("aspect_sentiment", "Aspect Sentiment", "aspect_sentiment.png",
     "Polarity split within each extracted product aspect."),
    ("review_length_distribution", "Review Length", "review_length_distribution.png",
     "Distribution of review length in tokens."),
    ("top_words", "Top Words", "top_words.png",
     "Highest-frequency terms after stopword removal."),
]
