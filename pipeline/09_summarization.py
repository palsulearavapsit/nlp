"""
PIPELINE STAGE 09 - TEXT SUMMARIZATION (extractive, TextRank)
=============================================================

WHAT THIS STAGE DOES
    Writes a short summary (the 3 most important sentences) for the first
    5,000 reviews; reviews under 40 words are stored whole.
    These 5,000 reviews become the "retrieval corpus" that the search and
    similarity features work on (stage 10 embeds exactly these rows).

INPUT
    amazon_reviews_topics.csv        (from stage 08)
OUTPUT
    amazon_reviews_summarized.csv    columns: title, review, summary,
                                              vader_sentiment, vader_score, topic

    The app reads this file whenever it shows a review from the retrieval
    corpus - see app/data.py -> review_at().

NLP CONCEPT - EXTRACTIVE vs ABSTRACTIVE SUMMARIZATION
    Extractive  - picks the most important sentences from the original text and
                  copies them unchanged. Nothing new is written, so a summary
                  can never state something the reviewer did not say.
    Abstractive - writes new sentences (needs a neural model such as T5/BART).
    This project uses EXTRACTIVE summarization.

NLP CONCEPT - TEXTRANK
    TextRank applies Google's PageRank algorithm to sentences:
        1. Each sentence is a node in a graph.
        2. Two sentences are joined by an edge weighted by how similar they are
           (how many words they share).
        3. PageRank scores every node: a sentence is important if it is similar
           to many other important sentences - it is "central" to the text.
        4. The top-N sentences are the summary.
    The app has its own commented TextRank implementation in app/summarize.py
    if you need to explain the maths step by step.

WHY ONLY 5,000 REVIEWS?
    The script does not record a reason. The likely one is speed: TextRank
    here and the embeddings in stage 10 are the slowest steps, and 5,000
    reviews keeps both fast enough for a laptop and a live demo.

RUN (from the project root)
    python pipeline/09_summarization.py
"""

import pandas as pd
import re

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_topics.csv")

# Use only 5,000 reviews for summarization
df = df.head(5000).copy()

summarizer = TextRankSummarizer()


def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_text(text, sentences=3):

    text = clean_text(text)

    # Short reviews don't need summarization
    if len(text.split()) < 40:
        return text

    try:
        # sumy splits the text into sentences and words (Tokenizer uses
        # NLTK's English sentence splitter) ...
        parser = PlaintextParser.from_string(
            text,
            Tokenizer("english")
        )

        # ... then TextRank returns the `sentences` highest-ranked sentences,
        # in their original order.
        summary = summarizer(
            parser.document,
            sentences
        )

        return " ".join(str(sentence) for sentence in summary)

    except Exception:
        return text


print("\nGenerating summaries for 5,000 reviews...")

df["summary"] = df["review"].apply(
    lambda x: summarize_text(x, 3)
)

# Save
df[
    [
        "title",
        "review",
        "summary",
        "vader_sentiment",
        "vader_score",
        "topic"
    ]
].to_csv(
    "amazon_reviews_summarized.csv",
    index=False
)

print("\nSaved:")
print("amazon_reviews_summarized.csv")

print("\nText summarization completed!")
