# The Review Room

**NLP intelligence for the Amazon review corpus.**

An interactive reading room over 100,000 Amazon customer reviews. Eleven NLP
stages turn unstructured review text into sentiment, product aspects, latent
topics, summaries and meaning-based retrieval, all shown through a single
editorial interface.

```bash
pip install -r requirements.txt
python run.py
```

Opens on <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.

> **Presenting or preparing for a viva?** Read **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**.
> It explains every stage, every NLP concept and every file, walks through a
> worked example, and ends with likely examiner questions and their answers.
> Inside the app, every page has a **How this works** panel under its title.

---

## What is this project?

Online stores collect far more reviews than anyone can read. This project uses
Natural Language Processing (NLP) to read them automatically and answer
practical questions:

| Question a business asks | NLP technique used | Page |
|---|---|---|
| Is this review positive or negative, and **why**? | Hybrid lexicon-based sentiment analysis (VADER + product-review rules) | 02 Verdict |
| Which product **aspects** (price, battery, shipping…) do customers like or hate? | Aspect-based sentiment analysis | 03 Aspects |
| What **themes** run through 100,000 reviews? | Topic modelling (TF-IDF + NMF) | 04 Topics |
| Find reviews that **mean** "it broke quickly", whatever words they use | Semantic search (sentence embeddings + cosine similarity) | 05 Retrieval |
| Which reviews say the **same thing** as this one? | Semantic similarity (nearest neighbours) | 05 Retrieval → *Find similar* |
| Give me the **gist** of this long review | Extractive summarization (TextRank) | 06 Condense |

## How it works (one picture)

The project has two halves. The **offline pipeline** did the heavy NLP once and
saved the results. The **web app** loads those results and runs three engines
live on whatever you type.

```
 OFFLINE (pipeline/, run once)                        ONLINE (app/ + web/, every click)
 ─────────────────────────────                        ─────────────────────────────────
 train.csv (3.6M raw reviews)                         browser (web/index.html + app.js)
   │ 01 sample 50k pos + 50k neg                              │  fetch /api/...
   ▼                                                          ▼
 amazon_reviews_100k.csv                              app/server.py  (FastAPI)
   │ 02-05 exploratory: preprocessing, n-grams,           │        │          │
   │       TF-IDF/RAKE keywords, NER (console only)       ▼        ▼          ▼
   │ 06 VADER sentiment ──► amazon_reviews_sentiment.csv ─┐   sentiment.py summarize.py semantic.py
   │ 07 aspect sentiment ─► amazon_review_aspects.csv  ───┤   (VADER +   (TextRank)  (MiniLM +
   │                        aspect_sentiment_summary.csv ─┤    rules)                cosine)
   │ 08 TF-IDF + NMF ─────► amazon_reviews_topics.csv ────┼──► app/data.py loads these CSVs
   │                        topic_keywords.csv ───────────┤    (lazily, cached in memory)
   │ 09 TextRank ─────────► amazon_reviews_summarized.csv ┤
   │ 10 MiniLM embeddings ► review_embeddings.csv ────────┘
   │ 11 semantic search demo (console)
   │ 12 evaluation: VADER vs hybrid engine (console)
```

## NLP concepts used, and where to find them

| Concept | Where it runs | File |
|---|---|---|
| Class-balanced sampling | offline | `pipeline/01_prepare_dataset.py` |
| Cleaning, tokenization, stopword removal, lemmatization | offline (exploratory) | `pipeline/02_preprocessing.py` |
| N-grams (unigrams, bigrams, trigrams) | offline (exploratory) | `pipeline/03_ngram_analysis.py` |
| TF-IDF, RAKE keyword extraction | offline (exploratory) | `pipeline/04_keyword_extraction.py` |
| Named Entity Recognition (spaCy) | offline (exploratory) | `pipeline/05_ner_analysis.py` |
| Lexicon-based sentiment (VADER) | offline | `pipeline/06_sentiment_analysis.py` |
| Aspect-based sentiment analysis | offline | `pipeline/07_aspect_sentiment.py` |
| Topic modelling: TF-IDF + NMF | offline | `pipeline/08_topic_extraction.py` |
| Extractive summarization (TextRank) | offline + **live** | `pipeline/09_summarization.py`, `app/summarize.py` |
| Sentence embeddings (all-MiniLM-L6-v2, 384-d) | offline | `pipeline/10_embeddings_similarity.py` |
| Cosine similarity, semantic search | offline demo + **live** | `pipeline/11_semantic_search.py`, `app/semantic.py` |
| Hybrid sentiment: VADER + rule lexicon, with evidence | **live** | `app/sentiment.py`, `app/lexicon.py` |
| Evaluation: accuracy, confusion matrix, precision/recall/F1 | offline | `pipeline/12_evaluate_sentiment.py` |

## Results at a glance

| Measurement | Value |
|---|---|
| Reviews analysed | 100,000 (50,000 positive + 50,000 negative) |
| VADER alone, agreement with human labels | **70.45%** |
| Hybrid engine (VADER + rules), agreement with human labels | **76.27%** |
| Human-negative reviews wrongly called positive | VADER 24,180 → hybrid 17,833 (26% fewer) |
| Topics found by NMF | 10 (from "Video games" to "Books & authors") |
| Product aspects tracked | 28 (only *customer service* has a net-negative score) |
| Reviews embedded for search | 5,000 × 384 dimensions |

---

## The six pages

Page 01 is the overview; pages 02–06 each demonstrate one NLP technique.

| # | Page | What it answers |
|---|--------|-----------------|
| 01 | **Corpus** | What does the whole corpus look like? Volume, polarity, model agreement, aspect and topic distribution, plus six charts of the full run. |
| 02 | **Verdict** | Is this review positive or negative, and *why*? The engine marks up the reviewer's own words with the phrases that moved the score, and lists the calculation step by step. |
| 03 | **Aspects** | What are customers actually judging? 28 product aspects, each with its own polarity split and excerpts. |
| 04 | **Topics** | What themes run through the corpus? Ten NMF topics, each with its keywords, a name and sample reviews. |
| 05 | **Retrieval** | Find reviews by meaning rather than keyword, then press *Find similar* on any review to see which others say the same thing differently. |
| 06 | **Condense** | Extractive TextRank summary with compression statistics. |

### The signature move

**Verdict** does not just return a label. It returns the evidence: every matched
phrase and keyword is highlighted inline in the review text (negative language
underlined in vermillion, positive in moss, single keywords dotted) alongside
the polarity components, the rule that decided the outcome, and the exact
arithmetic. The model shows its working.

---

## Project structure

```
run.py                   start here: launches the web server and opens a browser
requirements.txt         libraries the web app needs
PROJECT_GUIDE.md         the full explanation: concepts, walkthroughs, viva Q&A

pipeline/                OFFLINE: the scripts that computed the CSVs (run once, in order)
  01_prepare_dataset.py ... 12_evaluate_sentiment.py   (see pipeline/README.md)

app/                     ONLINE: the Python web application
  server.py              FastAPI routes: every /api/... endpoint, plus static hosting
  config.py              file paths, limits, topic names, chart captions
  data.py                loads the CSVs once and caches them (.npy cache for embeddings)
  sentiment.py           LIVE hybrid VADER + rule engine, with evidence and reasoning
  lexicon.py             122 phrases + 45 keywords tuned for product reviews
  summarize.py           LIVE TextRank summarisation
  semantic.py            LIVE embedding search and nearest neighbours
  plates.py              draws the six chart images in visuals/

web/                     the interface: plain HTML, CSS and JavaScript, no framework
  index.html             the six pages, each with a "How this works" panel
  assets/app.js          hash router + one controller per page (calls the API)
  assets/styles.css      design system

tests/                   pytest checks for the engines and the API
visuals/                 rendered chart images (bottom of the Corpus page)
legacy/                  the earlier Streamlit dashboard, kept for reference only
*.csv                    pipeline outputs (see "Data" below)
```

No build step, no frontend framework, no bundler. The interface is plain HTML,
CSS and ES modules served straight from `web/`.

### Endpoints

```
GET  /api/status                      which engines are live
GET  /api/overview                    corpus statistics + plate manifest
POST /api/sentiment                   { text }             -> verdict, scores, evidence, reasoning
POST /api/summarize                   { text, sentences }  -> summary + compression stats
GET  /api/aspects                     aspect ledger with polarity shares
GET  /api/aspects/{aspect}/reviews    excerpts for one aspect
GET  /api/topics                      topics with keywords, labels and polarity
GET  /api/topics/{topic}/reviews      sample reviews for one topic
POST /api/search                      { query, top_k }     -> ranked reviews
POST /api/similar                     { index, top_k }     -> nearest neighbours
GET  /api/reviews/{index}             one indexed review
GET  /api/reviews/random/pick         a random indexed review
```

---

## Graceful degradation

The three heavyweight NLP packages are optional. Without them the app still
runs, and the rail reports exactly which engine answered:

| Engine | With the package | Without it |
|--------|------------------|------------|
| Sentiment | `vaderSentiment`: hybrid VADER + rules | lexicon-only scoring |
| Summariser | `sumy` TextRank | built-in TextRank over word-overlap similarity |
| Retrieval | `sentence-transformers`: semantic search | IDF-weighted lexical ranking |
| Similarity | (none needed) | always available (uses stored embeddings) |

Install them for full fidelity:

```bash
pip install vaderSentiment sumy sentence-transformers
```

---

## Data

| File | Rows | Contents | Made by |
|------|------|----------|---------|
| `amazon_reviews_100k.csv` | 100,000 | title, review, human label | `pipeline/01` |
| `amazon_reviews_sentiment.csv` | 100,000 | + combined text, VADER label and compound score | `pipeline/06` |
| `amazon_review_aspects.csv` | 97,673 | one row per (aspect, review) pair | `pipeline/07` |
| `aspect_sentiment_summary.csv` | 28 | polarity counts per aspect | `pipeline/07` |
| `amazon_reviews_topics.csv` | 100,000 | reviews with NMF topic 1–10 | `pipeline/08` |
| `topic_keywords.csv` | 10 | top 10 NMF terms per topic | `pipeline/08` |
| `amazon_reviews_summarized.csv` | 5,000 | the retrieval corpus, with TextRank summaries | `pipeline/09` |
| `review_embeddings.csv` | 5,000 × 384 | MiniLM sentence embeddings (row *i* = review *i* above) | `pipeline/10` |

The embedding matrix is cached to `.cache/review_embeddings.npy` on first load,
so subsequent starts are instant.

## Tests

```bash
pip install pytest
python -m pytest
```

---

## Design

**Newsprint, ink, and one signal colour.** Warm paper ground (`#F0EBE1`), deep
ink (`#14130F`), vermillion accent (`#D8380F`), with moss and rust carrying
polarity. Fraunces for display, Archivo for interface, IBM Plex Mono for data
and labels. Hairline rules instead of card shadows; typography carries the
hierarchy.

The analytical plates in `visuals/` are re-rendered from the same source CSVs in
the same palette, so the charts belong to the interface rather than sitting
inside it:

```bash
python -m app.plates
```

Responsive from 320px up. Honours `prefers-reduced-motion`. Prints cleanly.

## Legacy

The original Streamlit dashboard is preserved at
`legacy/dashboard_streamlit.py`, and the original matplotlib plates at
`legacy/visuals-original/`. The sentiment lexicons and scoring logic were
carried over verbatim. The one difference: the old version collapsed extra
spaces and line breaks before matching phrases, the new one matches the text
as typed (so highlights line up exactly). No verdict in the 100k dataset
changes because of it.
