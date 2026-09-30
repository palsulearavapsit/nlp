"""
PIPELINE STAGE 05 - NAMED ENTITY RECOGNITION (NER)
==================================================

WHAT THIS STAGE DOES
    Runs spaCy's pre-trained NER model on the first 20 reviews, prints every
    entity it finds, and prints how many entities of each type were found.

    EXPLORATORY STAGE: it prints to the console and writes no file. Nothing in
    the app depends on it.

INPUT   amazon_reviews_100k.csv   (from stage 01)
OUTPUT  console only

NLP CONCEPT - NAMED ENTITY RECOGNITION
    NER finds spans of text that name real-world things and labels their type.
    Typical spaCy labels:
        PERSON   people              "Stephen King"
        ORG      organisations       "Amazon", "Sony"
        GPE      countries/cities    "China", "London"
        DATE     dates               "last Tuesday", "2005"
        CARDINAL numbers             "two"
        MONEY    amounts             "$20"
        PRODUCT  products            "iPod"
        WORK_OF_ART  titles          "Harry Potter"
    spaCy's "en_core_web_sm" is a small statistical model trained on
    annotated English text (OntoNotes). It predicts entities from the words and
    their context, so it can find names it has never seen before.

WHY IT IS USEFUL FOR REVIEWS
    Entities show which brands, authors, artists and places customers talk
    about - for example, reviews that name a competing brand.

RUN (from the project root)
    python -m spacy download en_core_web_sm     # once
    python pipeline/05_ner_analysis.py
"""

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

# Counts how many entities of each TYPE (PERSON, ORG, ...) were found.
entity_counter = Counter()

for i in range(20):

    text = df.loc[i, "text"]

    # NER runs on the ORIGINAL text: capital letters are a strong clue for
    # names, so this text is deliberately not lowercased or cleaned.
    doc = nlp(text)

    print("=" * 80)
    print("REVIEW:")
    print(text[:500])

    print("\nENTITIES:")

    # doc.ents holds the entity spans; each has .text and a type label .label_
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
