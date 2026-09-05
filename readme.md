# The Review Room

**NLP intelligence for the Amazon review corpus.**

An interactive reading room over 100,000 Amazon customer reviews. Eleven NLP
stages turn unstructured review text into sentiment, product aspects, latent
topics, summaries and meaning-based retrieval — surfaced through a single
editorial interface.

```bash
pip install -r requirements.txt
python run.py
```

Opens on <http://127.0.0.1:8000>.

---

## What it does

| # | Module | What it answers |
|---|--------|-----------------|
| 01 | **Corpus** | What does the whole corpus look like? Volume, polarity, model agreement, aspect and topic distribution. |
| 02 | **Verdict** | Is this review positive or negative — and *why*? The engine marks up the reviewer's own words with the phrases that moved the score. |
| 03 | **Aspects** | What are customers actually judging? 28 extracted product facets, each with its own polarity split and excerpts. |
| 04 | **Topics** | What themes run through the corpus? Ten unsupervised clusters, each readable. |
| 05 | **Retrieval** | Find reviews by meaning rather than keyword. |
| 06 | **Neighbours** | Given one review, which others say the same thing differently? |
| 07 | **Condense** | Extractive TextRank summary with compression statistics. |
| 08 | **Atlas** | Six analytical plates rendered from the full corpus run. |

### The signature move

**Verdict** does not just return a label. It returns the evidence: every matched
phrase and keyword is highlighted inline in the review text — negative language
underlined in vermillion, positive in moss, single keywords dotted — alongside
the polarity components and the rule that decided the outcome. The model shows
its working.

---

## Architecture

```
run.py                 launcher (uvicorn + browser)
app/
  server.py            FastAPI routes, static host
  config.py            paths, limits, plate manifest
  data.py              lazy cached corpus access (.npy embedding cache)
  sentiment.py         hybrid VADER + rule engine, with evidence spans
  lexicon.py           122 phrases + 45 keywords (product-review tuned)
  summarize.py         TextRank summarisation
  semantic.py          embedding search and nearest neighbours
  plates.py            re-renders the six analytical plates in the palette
web/
  index.html           the eight modules
  assets/styles.css    design system
  assets/app.js        hash router + view controllers
visuals/               rendered plates
legacy/                the previous Streamlit dashboard and its plates
```

No build step, no frontend framework, no bundler. The interface is plain
HTML, CSS and ES modules served straight from `web/`.

### Endpoints

```
GET  /api/status                      which engines are live
GET  /api/overview                    corpus statistics + plate manifest
POST /api/sentiment                   { text }
POST /api/summarize                   { text, sentences }
GET  /api/aspects                     aspect ledger with polarity shares
GET  /api/aspects/{aspect}/reviews    excerpts for one aspect
GET  /api/topics                      topic list with polarity
GET  /api/topics/{topic}/reviews      representative reviews
POST /api/search                      { query, top_k }
POST /api/similar                     { index, top_k }
GET  /api/reviews/{index}             one indexed review
GET  /api/reviews/random/pick         a random indexed review
```

---

## Graceful degradation

The three heavyweight NLP packages are optional. Without them the app still
runs, and the rail reports exactly which engine answered:

| Engine | With the package | Without it |
|--------|------------------|------------|
| Sentiment | `vaderSentiment` — hybrid VADER + rules | lexicon-only scoring |
| Summariser | `sumy` TextRank | built-in TextRank over TF similarity |
| Retrieval | `sentence-transformers` — semantic search | IDF-weighted lexical ranking |
| Similarity | — | always available (uses stored embeddings) |

Install them for full fidelity:

```bash
pip install vaderSentiment sumy sentence-transformers
```

---

## Design

**Newsprint, ink, and one signal colour.** Warm paper ground (`#F0EBE1`), deep
ink (`#14130F`), vermillion accent (`#D8380F`), with moss and rust carrying
polarity. Fraunces for display, Archivo for interface, IBM Plex Mono for data
and labels. Hairline rules instead of card shadows; typography carries the
hierarchy.

The analytical plates in `visuals/` are re-rendered from the same source CSVs in
the same palette, so the Atlas belongs to the interface rather than sitting
inside it:

```bash
python -m app.plates
```

Responsive from 320px up. Honours `prefers-reduced-motion`. Prints cleanly.

---

## The pipeline

```
Amazon reviews
  → dataset prep → text preprocessing
  → n-gram analysis · keyword extraction (TF-IDF, RAKE) · named entities
  → sentiment analysis
  → aspect-based sentiment
  → topic extraction
  → summarisation
  → embeddings (MiniLM, 384d)
  → semantic search
  → the interface
```

## Data

| File | Rows | Contents |
|------|------|----------|
| `amazon_reviews_sentiment.csv` | 100,000 | reviews with human label and VADER prediction |
| `amazon_reviews_topics.csv` | 100,000 | reviews with topic assignment |
| `amazon_review_aspects.csv` | — | one row per (aspect, review) pair |
| `aspect_sentiment_summary.csv` | 28 | polarity counts per aspect |
| `amazon_reviews_summarized.csv` | 5,000 | the retrieval corpus, with summaries |
| `review_embeddings.csv` | 5,000 × 384 | MiniLM sentence embeddings |

The embedding matrix is cached to `.cache/review_embeddings.npy` on first load,
so subsequent starts are instant.

---

## Legacy

The original Streamlit dashboard is preserved at
`legacy/dashboard_streamlit.py`, and the original matplotlib plates at
`legacy/visuals-original/`. The sentiment lexicons and scoring logic were
carried over verbatim, so verdicts are unchanged.
