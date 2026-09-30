# `pipeline/`: the offline NLP pipeline

These eleven scripts are where the NLP was **computed**. They were run once, in
order, and they produced the CSV files in the project root. A twelfth script,
added later, evaluates the sentiment engines against the human labels
(VADER alone 70.45% vs the app's hybrid engine 76.27%). The web app in
`app/` only **reads** those CSVs and adds a few live engines of its own. It
never runs these scripts.

Each script opens with a header that explains the stage, its input and output,
and the NLP concept it uses.

| # | Script | NLP technique | Reads | Writes | Used by the app? |
|---|--------|---------------|-------|--------|------------------|
| 01 | `01_prepare_dataset.py` | Sampling, class balancing | `train.csv` (raw, not in repo) | `amazon_reviews_100k.csv` | indirectly |
| 02 | `02_preprocessing.py` | Cleaning, tokenization, stopwords, lemmatization (spaCy) | 100k | console only | no (exploratory) |
| 03 | `03_ngram_analysis.py` | Unigram / bigram / trigram counts | 100k | console only | no (exploratory) |
| 04 | `04_keyword_extraction.py` | TF-IDF, RAKE | 100k | console only | no (exploratory) |
| 05 | `05_ner_analysis.py` | Named Entity Recognition (spaCy) | 100k | console only | no (exploratory) |
| 06 | `06_sentiment_analysis.py` | VADER lexicon-based sentiment | 100k | `amazon_reviews_sentiment.csv` | **yes**: Corpus page |
| 07 | `07_aspect_sentiment.py` | Aspect-based sentiment (aspect lexicon + VADER) | sentiment | `amazon_review_aspects.csv`, `aspect_sentiment_summary.csv` | **yes**: Aspects page |
| 08 | `08_topic_extraction.py` | Topic modelling: TF-IDF + NMF | sentiment | `amazon_reviews_topics.csv`, `topic_keywords.csv` | **yes**: Topics page |
| 09 | `09_summarization.py` | Extractive summarization (TextRank) | topics | `amazon_reviews_summarized.csv` (5,000 rows) | **yes**: every review shown on the Retrieval page |
| 10 | `10_embeddings_similarity.py` | Sentence embeddings (MiniLM), cosine similarity | summarized | `review_embeddings.csv` (5,000 × 384) | **yes**: Retrieval (search + find similar) |
| 11 | `11_semantic_search.py` | Semantic search | summarized + embeddings | console only | logic re-used live in `app/semantic.py` |
| 12 | `12_evaluate_sentiment.py` | Evaluation: accuracy, confusion matrix, precision / recall / F1 | sentiment | console only | measures the live engine (added later) |

The charts at the bottom of the Corpus page are drawn by `app/plates.py` (`python -m app.plates`)
from the CSVs above.

## Re-running

Run from the **project root**, because every path is relative to it:

```bash
pip install -r pipeline/requirements.txt
python -m spacy download en_core_web_sm
python pipeline/06_sentiment_analysis.py
```

Stages 06–10 **overwrite** the CSVs the app depends on. Stage 01 needs the raw
`train.csv`, which is not in the repository. After re-running stage 10, delete
the `.cache/` folder so the app reloads the new embeddings.
