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
        parser = PlaintextParser.from_string(
            text,
            Tokenizer("english")
        )

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