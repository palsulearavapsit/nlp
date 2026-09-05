import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import nltk

nltk.download("punkt")

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_100k.csv")

# Combine title + review
df["text"] = (
    df["title"].fillna("") + " " +
    df["review"].fillna("")
)

# Basic cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# METHOD 1: TF-IDF
# =====================================================

print("\nRunning TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

feature_names = vectorizer.get_feature_names_out()

print("\nTOP GLOBAL TF-IDF TERMS:")

scores = tfidf_matrix.sum(axis=0).A1

top_indices = scores.argsort()[-30:][::-1]

for i in top_indices:
    print(f"{feature_names[i]:30} {scores[i]:.2f}")


# =====================================================
# METHOD 2: RAKE
# =====================================================

print("\n\nRunning RAKE on sample reviews...")

rake = Rake()

for i in range(5):

    rake.extract_keywords_from_text(df.loc[i, "text"])

    phrases = rake.get_ranked_phrases_with_scores()

    print("\n" + "=" * 70)
    print("REVIEW:", df.loc[i, "text"][:300])

    print("\nTOP KEY PHRASES:")

    for score, phrase in phrases[:10]:
        print(f"{phrase:40} {score:.2f}")


print("\nKeyword extraction completed!")