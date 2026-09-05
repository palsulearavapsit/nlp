import pandas as pd
import spacy
from collections import Counter

print("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("Loading dataset...")

df = pd.read_csv("amazon_reviews_100k.csv")

df["text"] = (
    df["title"].fillna("") + " " +
    df["review"].fillna("")
)

# =====================================================
# NER ON SAMPLE REVIEWS
# =====================================================

print("\nRunning Named Entity Recognition...\n")

entity_counter = Counter()

for i in range(20):

    text = df.loc[i, "text"]

    doc = nlp(text)

    print("=" * 80)
    print("REVIEW:")
    print(text[:500])

    print("\nENTITIES:")

    for ent in doc.ents:

        print(f"{ent.text:30} → {ent.label_}")

        entity_counter[ent.label_] += 1


# =====================================================
# ENTITY SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("ENTITY TYPE SUMMARY")
print("=" * 80)

for entity_type, count in entity_counter.most_common():

    print(f"{entity_type:20} {count}")


print("\nNER analysis completed!")