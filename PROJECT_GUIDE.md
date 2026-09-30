# Project Guide: The Review Room

This guide explains the whole project: what it does, how every part works, the
NLP behind it, the results, and the questions an examiner is likely to ask.
Read it top to bottom once. After that, use the contents list to jump.

Every claim here points at a file, so you can open the code and show it. Every
number here was measured on this project's own data.

## Contents

1. [The project in 60 seconds](#1-the-project-in-60-seconds)
2. [The big picture: two halves](#2-the-big-picture-two-halves)
3. [The dataset](#3-the-dataset)
4. [The offline pipeline, stage by stage](#4-the-offline-pipeline-stage-by-stage)
5. [The web application](#5-the-web-application)
6. [Each page, step by step](#6-each-page-step-by-step)
7. [NLP concepts explained](#7-nlp-concepts-explained)
8. [Results and what they mean](#8-results-and-what-they-mean)
9. [Viva questions and answers](#9-viva-questions-and-answers)
10. [Limitations and possible improvements](#10-limitations-and-possible-improvements)
11. [Five-minute demo script](#11-five-minute-demo-script)
12. [Quick reference: where is it in the code?](#12-quick-reference-where-is-it-in-the-code)

---

## 1. The project in 60 seconds

> Online stores collect far more customer reviews than anyone can read. Our
> project, **The Review Room**, uses NLP to read 100,000 Amazon reviews
> automatically. It tells you whether a review is positive or negative and
> *shows why*, which product aspects (price, battery, shipping…) customers
> praise or criticise, what the main themes of the reviews are, lets you search
> reviews by *meaning* instead of exact words, finds reviews that say the same
> thing, and summarizes long reviews.
>
> It has two parts. An **offline pipeline** of twelve Python scripts did the
> heavy NLP once (VADER sentiment, aspect extraction, TF-IDF + NMF topic
> modelling, TextRank summarization, MiniLM sentence embeddings) and saved the
> results. A **web application** (FastAPI backend + HTML/JavaScript frontend)
> presents those results and runs three NLP engines live: a hybrid sentiment
> analyser, a TextRank summarizer and a semantic search engine.
>
> Our hybrid sentiment engine agrees with the human labels on **76.27%** of the
> reviews, up from **70.45%** for VADER alone.

---

## 2. The big picture: two halves

```
        OFFLINE: pipeline/                          ONLINE: app/ + web/
        (run once, takes minutes to hours)          (runs on every click, milliseconds)

  raw reviews ──► 12 Python scripts ──► CSV files ──► app/data.py loads them
                                                            │
                                                            ▼
                                        browser ◄──JSON── app/server.py ◄── live engines:
                                     (web/index.html,                        sentiment.py
                                      web/assets/app.js)                     summarize.py
                                                                             semantic.py
```

**Why split it this way?** Some NLP steps are slow. Running VADER over 100,000
reviews, fitting a topic model or computing 5,000 Transformer embeddings takes
minutes, so you do it once and save the result. Other steps are fast enough to
run instantly on text a user types, such as scoring one review or summarizing
one paragraph. The app does only the fast work live and reads the slow work
from disk.

| | Offline pipeline | Online app |
|---|---|---|
| Folder | `pipeline/` | `app/` (Python), `web/` (browser) |
| Runs | once, by hand, in order | every time you click |
| Input | raw review CSV | the pipeline's CSVs + what you type |
| Output | CSV files in the project root | JSON to the browser, shown as pages |
| NLP | all techniques | hybrid sentiment, TextRank, semantic search |

---

## 3. The dataset

- **Source:** the public *Amazon Review Polarity* dataset (Zhang, Zhao & LeCun,
  2015), commonly downloaded from Kaggle. Its `train.csv` has about 3.6 million
  reviews with three columns and no header: `label` (1 = negative,
  2 = positive), `title`, `review`.
- **How labels were made (by the dataset's authors):** 1–2 star reviews are
  *negative*, 4–5 star reviews are *positive*, and 3-star reviews were dropped.
  So **there is no "neutral" human label.**
- **Our working set:** `pipeline/01_prepare_dataset.py` took the first 50,000
  positive and 50,000 negative reviews and shuffled them
  → `amazon_reviews_100k.csv`.
- **Why balanced?** With 50/50 classes, always guessing "positive" scores only
  50%, so the accuracy we report is meaningful.
- **What the reviews are about:** despite "Amazon products", most are about
  **books, movies and music**. Once stopwords are removed, the most frequent
  word is *book* (61,967 times), then *movie* (31,239) (stage 03 counts).
- **Length:** median 67 words per review (mean 75.6; 99% are under 176 words).

---

## 4. The offline pipeline, stage by stage

Each script in `pipeline/` begins with a header explaining its stage. Stages
02–05 are **exploratory**: they print results to the console and save nothing.
Stages 06–10 **produce the files the app reads**.

### Stage 01: dataset preparation (`01_prepare_dataset.py`)
Streams the huge raw file in chunks of 100,000 rows, keeps 50,000 of each
class, maps 1/2 → negative/positive and shuffles (`random_state=42`).
**Output:** `amazon_reviews_100k.csv`.

### Stage 02: text preprocessing (`02_preprocessing.py`, exploratory)
Shows the classic cleaning chain on 10 reviews, printing each one before and
after: remove HTML tags and URLs → lowercase → keep letters only → spaCy
tokenization → drop stopwords → **lemmatize** (`batteries → battery`,
`was → be`).
Later stages don't call this script. Each one applies the cleaning it needs:
regex cleaning plus scikit-learn's built-in stopword list.

### Stage 03: n-gram analysis (`03_ngram_analysis.py`, exploratory)
`CountVectorizer` counts every unigram, bigram and trigram (stopwords removed)
across all 100k reviews. Real results:

| Top unigrams | Top bigrams | Top trigrams |
|---|---|---|
| book 61,967 | read book 3,930 | don waste money 734 |
| movie 31,239 | **waste money 2,040** | don waste time 556 |
| like 30,195 | year old 1,933 | waste time money 416 |
| good 30,191 | waste time 1,807 | book ve read 266 |
| great 29,973 | book read 1,778 | **highly recommend book** 237 |

"don" and "ve" are what is left of *don't* and *I've* once punctuation is
removed: the apostrophe goes, and the one-letter "t" is dropped. The strong
phrases here, like "waste money" and "highly recommend", are exactly the kind
the live sentiment engine's phrase lexicon (`app/lexicon.py`) looks for.

### Stage 04: keyword extraction (`04_keyword_extraction.py`, exploratory)
Two methods:
- **TF-IDF** over the whole corpus. Top terms: book, movie, great, good, read,
  like, just, time, really, love…
- **RAKE** inside single reviews. For *"Amazing This book is a must read for
  women of all ages!! Full of valuable lessons, the best book I've read in a
  while!"* RAKE returns: *valuable lessons* (4.0), *must read* (3.5), *best
  book* (3.5).

### Stage 05: named entity recognition (`05_ner_analysis.py`, exploratory)
spaCy's pre-trained `en_core_web_sm` model tags entities (PERSON, ORG, GPE,
DATE, …) in 20 reviews and counts the entity types. NER runs on the original
text on purpose, because capital letters are strong clues for names.

### Stage 06: sentiment analysis with VADER (`06_sentiment_analysis.py`)
VADER scores `title + " " + review` for all 100k reviews. Label rule:
compound ≥ 0.05 → positive, ≤ −0.05 → negative, otherwise neutral.
**Output:** `amazon_reviews_sentiment.csv` (adds `text`, `vader_sentiment`,
`vader_score`). **Result:** 70.45% agreement with the human labels.

### Stage 07: aspect-based sentiment (`07_aspect_sentiment.py`)
A fixed list of **28 aspect words** (price, quality, battery, performance,
design, size, sound, camera, screen, display, comfort, durability, material,
packaging, shipping, delivery, service, customer service, seller, product,
book, story, movie, music, audio, video, software, value). Each review is
searched for each word as a whole word (`\bprice\b`). Every aspect found gets
the review's VADER label. **Outputs:** `amazon_review_aspects.csv` (97,673
aspect–review pairs) and `aspect_sentiment_summary.csv` (28 rows of counts).

### Stage 08: topic extraction with TF-IDF + NMF (`08_topic_extraction.py`)
Clean → TF-IDF (unigrams + bigrams, 10,000 terms, `min_df=5`, `max_df=0.9`)
→ **NMF with 10 topics** (`random_state=42`, `init="nndsvda"`, 200 iterations)
→ each review gets its strongest topic (argmax, numbered 1–10).
**Outputs:** `amazon_reviews_topics.csv`, `topic_keywords.csv`. See the topic
table in [section 8](#8-results-and-what-they-mean).

### Stage 09: summarization with TextRank (`09_summarization.py`)
For the first **5,000** reviews: reviews of 40+ words get a 3-sentence TextRank
summary via the `sumy` library; shorter ones are kept as they are.
**Output:** `amazon_reviews_summarized.csv`. These 5,000 reviews are
the *retrieval corpus* used by search and similarity.

### Stage 10: sentence embeddings (`10_embeddings_similarity.py`)
Encodes the `review` text of the 5,000 reviews with the **all-MiniLM-L6-v2**
sentence-transformer into 384-number vectors, and demonstrates cosine
similarity. **Output:** `review_embeddings.csv` (5,000 × 384; row *i* belongs
to review *i* of the summarized file).

### Stage 11: semantic search demo (`11_semantic_search.py`)
Embeds three example queries and prints the five most similar reviews for
each. The live version of this logic is `app/semantic.py → search()`.

### Stage 12: evaluation (`12_evaluate_sentiment.py`, added later)
Runs the live hybrid engine over all 100k reviews and compares it with VADER
alone: accuracy, confusion matrix, precision / recall / F1. Results are in
[section 8](#8-results-and-what-they-mean).

---

## 5. The web application

### 5.1 How a click becomes a result

Example: you type a review on the **Verdict** page and press *Analyse*.

1. **Browser.** `web/assets/app.js` → the `verdict` controller catches the form
   submit and calls `api('/api/sentiment', { body: { text } })`, i.e. an HTTP
   `POST` with JSON `{"text": "..."}`.
2. **Server.** `app/server.py → analyze_sentiment()` receives it. FastAPI and
   pydantic check the body first (text must be 1–20,000 characters; otherwise
   HTTP 422).
3. **NLP.** It calls `app/sentiment.py → analyze(text)`, which runs the four
   steps in [section 6.2](#62-verdict-hybrid-sentiment-analysis) and returns a
   Python dict.
4. **Response.** FastAPI converts the dict to JSON and sends it back.
5. **Render.** `app.js → renderVerdict()` builds the result panel: the verdict
   banner, the highlighted text (`highlightEvidence()`), the score readout, the
   evidence chips and the "How the verdict was reached" steps.

Every page follows the same pattern: **app.js controller → `/api/...` endpoint
in server.py → Python NLP module → JSON → HTML**. The map:

| Page | JS controller | Endpoint | Python |
|---|---|---|---|
| 01 Corpus | `corpus` | `GET /api/overview` + `/visuals/*.png` | `server.py overview()` reads pipeline CSVs; charts drawn by `plates.py` |
| 02 Verdict | `verdict` | `POST /api/sentiment` | `sentiment.py analyze()` |
| 03 Aspects | `aspects` | `GET /api/aspects`, `/api/aspects/{a}/reviews` | `server.py aspect_table()`, `aspect_reviews()` |
| 04 Topics | `topics` | `GET /api/topics`, `/api/topics/{n}/reviews` | `server.py topic_list()`, `topic_reviews()` |
| 05 Retrieval | `retrieval` + `similarReviews` | `POST /api/search`, `POST /api/similar` | `semantic.py search()`, `neighbours()` |
| 06 Condense | `condense` | `POST /api/summarize` | `summarize.py summarize()` |

FastAPI also generates interactive docs for every endpoint at
<http://127.0.0.1:8000/docs>. You can try each endpoint there.

### 5.2 Every file and what it is for

| File | Purpose |
|---|---|
| `run.py` | Entry point. Starts the uvicorn web server with `app.server:api` and opens the browser. |
| `app/server.py` | All HTTP routes (`/api/...`), request validation, serving the web page and chart images. |
| `app/config.py` | Paths of every data file (with the pipeline stage that made it), limits, the embedding model name, hand-written topic names, chart captions. |
| `app/data.py` | The only module that reads files. Loads each CSV the first time it is needed and caches it; converts the embeddings CSV into a fast `.npy` file. |
| `app/sentiment.py` | **Live** hybrid sentiment engine: rule lexicon + VADER + decision rules, returning evidence spans and plain-English reasoning. |
| `app/lexicon.py` | The rule lexicon: 74 negative phrases, 48 positive phrases, 26 negative keywords, 19 positive keywords. |
| `app/summarize.py` | **Live** TextRank summarizer (sumy's, with a built-in fallback whose maths is fully commented). |
| `app/semantic.py` | **Live** semantic search and nearest neighbours: cosine similarity over the 5,000 × 384 embedding matrix; lexical fallback. |
| `app/plates.py` | Draws the six chart images in `visuals/` (`python -m app.plates`). |
| `web/index.html` | The single HTML page holding all six views, each with a "How this works" panel. |
| `web/assets/app.js` | Router + one controller per view. Calls the API and renders results. No NLP runs in the browser. |
| `web/assets/styles.css` | The visual design. |
| `pipeline/01…12` | The offline scripts ([section 4](#4-the-offline-pipeline-stage-by-stage)). |
| `tests/` | pytest checks that pin down the documented behaviour. |
| `legacy/` | The earlier Streamlit version of the dashboard, kept for reference only. |

### 5.3 Design decisions worth knowing

- **Lazy loading + caching** (`app/data.py`). The CSVs total about 260 MB.
  Each is loaded the first time a feature needs it, then kept in memory. A lock
  stops two simultaneous requests from loading the same file twice. (The
  engine-status box in the left rail runs on the first page view, so the
  embeddings and the 5,000 summaries load straight away.)
- **Graceful degradation.** `vaderSentiment`, `sumy` and
  `sentence-transformers` are optional. If one is missing, a simpler built-in
  engine answers instead. The left rail of the app shows which engine is live
  (green = full, amber = fallback).
- **No frontend framework.** Plain HTML, CSS and JavaScript modules, with no
  build step.

---

## 6. Each page, step by step

### 6.1 Corpus: overview

**What you see:** totals for the whole 100k corpus, two polarity bars (human
labels vs VADER), the 11 pipeline stages, the biggest aspects and topics, and
six charts at the bottom.

**What happens:** `GET /api/overview` → `server.py overview()` counts values in
the pipeline CSVs. No model runs.

- **Human labels** bar: exactly 50,000 / 50,000, no neutral.
- **VADER prediction** bar: 70,374 positive, 2,131 neutral, 27,495 negative.
  VADER leans positive.
- **Model agreement 70.5%**: the share of reviews where VADER's label equals
  the human label, i.e. VADER's accuracy.
- **Charts:** `app/plates.py` read the pipeline CSVs and drew six PNG charts
  with matplotlib (sentiment distribution, topic distribution, aspect
  frequency, aspect sentiment, review length, top words). The page only
  displays them; click one to enlarge it. Each caption says what the chart
  counts and which pipeline stage produced the data.

### 6.2 Verdict: hybrid sentiment analysis

**Code:** `app/sentiment.py → analyze()`, word lists in `app/lexicon.py`.

**The four steps:**

1. **Rule lexicon.** Look for 122 phrases (matched anywhere in the text) and
   45 keywords (matched as whole words):
   `rule score = 0.40 × positive phrases + 0.08 × positive keywords − 0.50 × negative phrases − 0.10 × negative keywords`
2. **VADER.** Get the compound score (−1…+1) and the positive / negative /
   neutral proportions.
3. **Combine.**
   - only negative phrases found → **Negative**, `final = min(−0.30, compound + rule)`
   - only positive phrases found → **Positive**, `final = max(+0.30, compound + rule)`
   - otherwise (no phrases, or both kinds) → `final = 0.70 × compound + 0.30 × rule`;
     ≥ +0.05 Positive, ≤ −0.05 Negative, in between Neutral
4. **Hard rules.** If the text contains "very/too/extremely slow" (→ −0.60),
   "should not be bought / should not buy / do not buy / don't buy" (→ −0.80) or
   "waste of money / waste money" (→ −0.80), the verdict is forced to
   Negative with that score. The rules are checked in that order, so if several
   match, the one listed last sets the score, wherever the phrases sit in
   the text.

**Worked example.** *"Not worth it. The battery is good but the screen broke
after a week."*

| Step | What happens | Number |
|---|---|---|
| 1 | Negative phrases: "not worth", "broke". Positive keyword: "good". Rule = 0.08×1 − 0.50×2 | **−0.920** |
| 2 | VADER: pos 0.108, neg 0.280, neu 0.612 | compound **−0.474** |
| 3 | Only negative phrases → override: min(−0.30, −0.474 + (−0.920)) | **−1.394** |
| 4 | No hard-rule phrase present | (unchanged) |
| | **Verdict** | **Negative, −1.394** |

The page shows exactly this: highlighted words in the review, the five
numbers, the evidence chips, and these steps in plain English.

**An example where the rules overrule VADER:** *"Very slow shipping but the
product itself is amazing."* VADER alone gives +0.735 (positive), because
"amazing" is strong and VADER weights the part after "but" more. The blend
gives +0.479, still positive. Then the hard rule for "very slow" forces
**Negative (−0.60)**. For a shopper, "very slow" is a decisive complaint.

**Reading the numbers:**
- *Positive / Negative / Neutral*: VADER's estimate of how much of the text is
  positive, negative or neutral. They add up to 1.
- *Compound*: VADER's overall score, −1 to +1.
- *Rule score*: the lexicon's score (can go beyond ±1).
- *Final score*: after the combination. Under a phrase override it is
  compound + rule, so it **can exceed ±1** (e.g. −4.2 for a review full of
  complaints). Its sign and size matter, not a fixed range.

### 6.3 Aspects: aspect-based sentiment

**Code:** computed offline by `pipeline/07_aspect_sentiment.py`; served by
`server.py aspect_table()` and `aspect_reviews()`.

**What happens:** the page loads the 28-row summary. For each aspect it shows
the mention count, the positive / neutral / negative split, and
**net sentiment = (positive − negative) ÷ total**. Clicking an aspect fetches
four sample reviews that mention it (a fixed sample from the first 400, so the
same ones appear on every visit).

**Things to point out:**
- *Customer service* is the **only aspect with a negative net score (−2%)**.
- *Battery* is the weakest product feature (+15%). *Comfort* is the best-loved
  (+76%).
- Limitation: the label comes from the whole review, not the sentence about
  the aspect (see [section 10](#10-limitations-and-possible-improvements)).

### 6.4 Topics: topic modelling

**Code:** computed offline by `pipeline/08_topic_extraction.py`; served by
`server.py topic_list()` and `topic_reviews()`; topic names in
`app/config.py TOPIC_LABELS`.

**What happens:** each topic chip shows its number, a short name and its size.
Selecting one shows its **top 10 NMF keywords**, the VADER polarity of its
reviews, and 8 sample reviews whose strongest topic it is (a fixed sample from
the first 500).

The names were **written by hand** after reading the keywords. NMF itself only
outputs numbered topics. Labelling topics by hand is standard practice.

### 6.5 Retrieval: semantic search and similar reviews

This page has two parts that use the same idea: every review is a point in
"meaning space", and closeness is measured with cosine similarity.

**Search** (`app/semantic.py → search()`):

1. The query is encoded by **all-MiniLM-L6-v2**, the same model that embedded
   the 5,000 reviews, into a 384-number vector.
2. **Cosine similarity** between the query vector and all 5,000 review vectors,
   computed in one matrix multiplication (`matrix @ query / norms`).
3. The top *k* reviews are returned, best first. Each card shows the review's
   stored TextRank summary, its similarity score and a **Find similar →**
   button.

**Fallback:** without `sentence-transformers`, the search runs in **lexical**
mode: cosine similarity between IDF-weighted word-count vectors. It finds
reviews that share the query's words, with rarer words counting more. The
results header states which mode answered.

**Find similar reviews** (`app/semantic.py → neighbours()`): type a review
index, press **Random**, or click **Find similar →** on any result. The code
takes the stored embedding of review *i* (row *i* of the matrix), computes its
cosine similarity with all 5,000 embeddings, drops review *i* itself (it always
scores 1.0) and returns the top *k*. **No model is needed** because every
vector was computed offline, so this works even without
`sentence-transformers`.

### 6.6 Condense: extractive summarization

**Code:** `app/summarize.py → summarize()`.

The idea (TextRank):

1. Split the review into sentences.
2. Turn each sentence into a list of words.
3. Build a graph: sentences are nodes, and the more words two sentences share,
   the stronger the edge between them.
4. Run **PageRank** (damping 0.85). A sentence scores high when it is similar
   to many other high-scoring sentences.
5. Keep the top *N* sentences, printed in their original order.

There are **two implementations** of this idea:

| | Live engine: `sumy`'s TextRank | Built-in fallback (only without sumy) |
|---|---|---|
| Stopwords | kept | removed |
| Edge weight | shared words ÷ (log\|A\| + log\|B\|) | shared words ÷ (√\|A\| × √\|B\|) |
| PageRank | repeated until the scores stop changing | 40 fixed rounds |
| Code | the `sumy` library (same one as pipeline stage 09) | written out in `_textrank()` and `_sentence_similarity()` |

(|A| = number of words in sentence A.) The built-in PageRank update is
`score(i) = 0.15/n + 0.85 × Σ_j [ w(j,i) / Σ_k w(j,k) × score(j) ]`.

Reviews under 25 words, or with no more sentences than requested, come back
unchanged. **Compression** = 1 − summary words ÷ original words, i.e. the share
of the text removed.

---

## 7. NLP concepts explained

**Corpus.** The collection of texts being studied: here, 100,000 reviews.

**Tokenization.** Splitting text into units (tokens), usually words.
"Don't buy it!" → `don't`, `buy`, `it`, `!`. Used everywhere, e.g. the
regexes in `summarize.py` and `semantic.py`, and scikit-learn's vectorizers.

**Stopwords.** Very common words ("the", "is", "and") that carry little meaning
on their own. They are removed for counting, topic modelling and the built-in
TextRank, so
they don't dominate. They are deliberately **kept** for VADER ("not good"
needs "not") and NER (sentence structure helps).

**Stemming vs lemmatization.** Both reduce words to a base form. Stemming chops
suffixes by rule (`studies → studi`). Lemmatization uses a dictionary and the
part of speech to return a real word (`studies → study`, `was → be`).
Stage 02 uses spaCy lemmatization.

**N-grams.** Runs of *n* consecutive words: unigrams ("waste"), bigrams
("waste money"), trigrams ("don waste money"). They capture phrases whose
meaning comes from the words together. Stage 03.

**Bag of words / document-term matrix.** Each document becomes a vector of word
counts, with word order ignored. One row per document, one column per word.
`CountVectorizer` builds it (stage 03).

**TF-IDF.** Term Frequency × Inverse Document Frequency. A word gets a high
weight in a document if it is frequent **there** but rare **across** the
corpus: `idf(t) = log(N / df(t))`. So "the" scores near zero and "blu-ray"
scores high. Used in stage 04 (keywords), stage 08 (input to NMF) and the
lexical search fallback.

**RAKE.** Rapid Automatic Keyword Extraction. It splits one document into
candidate phrases at stopwords and punctuation, then scores each word by
*degree ÷ frequency* and each phrase by the sum of its word scores. It needs
no corpus. Stage 04.

**Named Entity Recognition (NER).** Finding and classifying names of things:
PERSON, ORG, GPE (places), DATE, MONEY, PRODUCT… spaCy uses a statistical
model trained on annotated text. Stage 05.

**Sentiment analysis.** Deciding whether a text expresses a positive,
negative or neutral opinion.
- *Lexicon-based* (VADER, our rules): look words up in rated lists and apply
  rules. No training data, and fully explainable.
- *Machine-learning-based*: a classifier trained on labelled examples. Usually
  more accurate, but harder to explain.

**VADER.** A lexicon-based sentiment tool with 7,506 rated entries
("good" +1.9, "horrible" −2.5) plus rules for negation, intensifiers ("very"),
exclamation marks, CAPITALS and "but" (the clause after "but" counts more). The
**compound score** is the normalised sum, from −1 to +1. Standard cut-offs:
±0.05.

**Hybrid sentiment.** Combining two methods so each covers the other's gaps.
Here: VADER (general language) + a product-review rule lexicon
(shopping-specific complaints).

**Aspect-based sentiment analysis (ABSA).** Sentiment per aspect ("battery:
positive, screen: negative") instead of per review. Two sub-tasks: aspect
extraction (ours: a 28-word lexicon) and aspect sentiment (ours: the review's
VADER label).

**Topic modelling.** Unsupervised discovery of themes: groups of words that
tend to appear together. Each document is treated as a mix of topics.

**NMF (Non-negative Matrix Factorization).** Approximates the documents × terms
TF-IDF matrix **X** as **W × H**. **W** (documents × k) says how much of each
topic every document contains. **H** (k × terms) says how important each term
is to each topic. All values are ≥ 0, so topics *add up* and stay readable. The
top-weighted terms in a row of H are that topic's keywords.

**LDA (Latent Dirichlet Allocation).** The other classic topic model. It is
probabilistic: topics are probability distributions over words, usually fitted
on raw counts. NMF is algebraic, works directly on TF-IDF, and is fast and
often cleaner on short texts.

**Supervised vs unsupervised.** Supervised learning trains on labelled examples.
Unsupervised learning finds structure without labels. Our NMF topic model is
unsupervised. None of our sentiment methods is trained. The human labels are
used only to **evaluate**.

**Extractive vs abstractive summarization.** Extractive: copy the most
important sentences (ours). Abstractive: generate new sentences with a neural
model (e.g. BART, T5).

**TextRank / PageRank.** PageRank ranks web pages by the links pointing to them,
weighted by the rank of the linking pages. TextRank applies this to sentences,
with similarity as the "link". The **damping factor** (0.85) is the probability
of following a link rather than jumping to a random node. It keeps scores
stable and ensures every sentence gets a little.

**Word / sentence embeddings.** Dense vectors that place texts in a space where
distance reflects meaning. Sentences with similar meaning land close together
even with no shared words.

**Transformer / MiniLM.** A Transformer reads all words at once and uses
*attention* to let each word's representation depend on its context.
**all-MiniLM-L6-v2** is a small 6-layer Transformer fine-tuned on about 1
billion sentence pairs to put paraphrases close together. It outputs 384
numbers per text (mean of the token vectors) and reads up to 256 word-pieces.

**Cosine similarity.** `cos(a, b) = a·b / (|a| |b|)`: the cosine of the angle
between two vectors. 1 = same direction (same meaning), 0 = unrelated. It
ignores vector length, so a long and a short review with the same meaning
still match.

**Semantic vs lexical search.** Lexical search matches words, so "broken"
won't find "stopped working". Semantic search matches meaning via embeddings.

**Evaluation metrics.**
- *Accuracy*: the share of predictions that are correct (our "agreement").
- *Confusion matrix*: a table of true label × predicted label that shows which
  mistakes happen.
- *Precision*: of the reviews predicted negative, the share that really are
  negative.
- *Recall*: of the reviews that really are negative, the share that were
  found.
- *F1*: the harmonic mean of precision and recall.

---

## 8. Results and what they mean

### 8.1 Sentiment: VADER alone vs our hybrid engine

Measured on all 100,000 reviews by `pipeline/12_evaluate_sentiment.py`:

| | Accuracy | Negative precision | Negative recall | Negative F1 | Positive F1 |
|---|---|---|---|---|---|
| VADER alone | 70.45% | 0.882 | 0.485 | 0.626 | 0.768 |
| **Hybrid engine** | **76.27%** | 0.888 | **0.615** | **0.727** | **0.803** |

Confusion matrices (rows = human label, columns = prediction):

| VADER alone | pred negative | pred neutral | pred positive |
|---|---|---|---|
| human negative | 24,251 | 1,569 | **24,180** |
| human positive | 3,244 | 562 | 46,194 |

| Hybrid | pred negative | pred neutral | pred positive |
|---|---|---|---|
| human negative | 30,770 | 1,397 | **17,833** |
| human positive | 3,889 | 610 | 45,501 |

**What this means:** VADER misses complaints. It called nearly half of the
negative reviews positive (negative recall 0.485). Real examples from the data
that VADER scored strongly positive although the customer gave 1–2 stars:

- *"Would have expected better! … the letters are rubbing off. For the price I
  paid that was dissapointing. Other than that I love the concept…"*: VADER
  +0.87
- *"so-so I've always enjoyed vampire movies… Though its cheesy, i still like
  it."*: VADER +0.83

With the rule lexicon, 6,347 fewer complaints are called positive
(24,180 → 17,833), at the cost of 645 more positive reviews being called
negative (3,244 → 3,889). Overall accuracy rises by 5.8 points.

**Why not higher?** Some reviews are subtle or mixed ("so-so… I still like
it"), and a 2-star review can sound polite. Lexicon methods, VADER and our
rules alike, cannot read that intent. Fine-tuned Transformer classifiers
reach much higher accuracy on this dataset (published results are above 95%),
but they need training and cannot show their reasoning the way our engine
does.

### 8.2 Topics found by NMF

| # | Name (hand-written) | Top NMF keywords | Reviews | Human-negative share |
|---|---|---|---|---|
| 1 | Novels & stories | read, books, story, characters, read book, reading, novel, series, time, life | 7,448 | 40% |
| 2 | Books & authors | book, reading, author, written, read book, great book, book book, information, pages, recommend book | 17,867 | 52% |
| 3 | Movies | movie, movies, watch, seen, bad, film, like, acting, funny, just | 11,958 | 54% |
| 4 | Products & money | product, use, just, don, money, work, time, buy, bought, waste | 26,638 | **67%** |
| 5 | "Great" praise | great, love, great book, price, great movie, works, works great, book great, great product, easy | 7,695 | **14%** |
| 6 | Music CDs & singers | cd, music, songs, song, love, clay, voice, listen, like, buy | 4,810 | 39% |
| 7 | DVD & Blu-ray editions | dvd, film, version, quality, ray, video, blu, blu ray, original, vhs | 6,790 | 55% |
| 8 | Albums & bands | album, songs, music, song, like, best, band, albums, rock, just | 6,798 | 38% |
| 9 | "Good" / lukewarm | good, really, pretty, good book, like, really good, pretty good, bad, price, quality | 7,906 | 39% |
| 10 | Video games | game, play, fun, games, graphics, like, played, playing, great game, buy | 2,090 | 52% |

**What this means:**
- Most topics are **subjects** (books, movies, music, games), which matches
  what the dataset contains.
- Topics 4, 5 and 9 are **opinion styles** rather than subjects. Topic 4 is the
  complaint topic (money, waste, don't buy), with 67% negative reviews. Topic 5
  gathers short enthusiastic reviews and is 86% positive. NMF groups words that
  co-occur, and sentiment words co-occur too.
- "clay" in topic 6 is mostly the singer **Clay Aiken**, named in 147 reviews
  of that topic.
- "don" is the remains of "don't" after punctuation removal.
- Re-running stage 08's exact configuration reproduced the stored topic of
  **99.9%** of reviews, so the keywords shown are the real model's.

### 8.3 Aspects

The largest aspects are book (27,970 mentions), movie (15,131), story (9,838),
product (6,640) and music (6,304). By net sentiment, the best-loved are
comfort (+76%), music (+63%), price (+62%) and value (+61%). The weakest are
**customer service (−2%)**, battery (+15%), seller (+23%) and service (+25%).
Customers are much happier with *what* they bought than with the *service
around it*.

### 8.4 Summaries

Of the 5,000 stored summaries, 2,063 (41%) are identical to the original
review. In 1,267 cases the review was under 40 words, so it was never
summarized. The other 796 reviews had three sentences or fewer, so "the top
three sentences" was the whole review. The median review is only 67 words.

---

## 9. Viva questions and answers

**Q: What is your project, in one sentence?**
An NLP system that analyses 100,000 Amazon reviews for sentiment, aspects,
topics, summaries and meaning-based search, with an interactive web interface
that explains its results.

**Q: What problem does it solve?**
Reading thousands of reviews by hand is impossible. The system shows a business
at a glance what customers feel, about which aspects and on which themes, and
lets it search reviews by meaning.

**Q: Which dataset did you use and why 100,000 balanced reviews?**
The Amazon Review Polarity dataset. 50k positive + 50k negative, so accuracy is
meaningful (a trivial model gets 50%) and processing fits on a laptop.
`pipeline/01_prepare_dataset.py`.

**Q: Where do the labels come from?**
From star ratings, set by the dataset's authors: 1–2 stars negative, 4–5
positive, 3 stars removed. That is why there is no neutral label.

**Q: What preprocessing did you do?**
Lowercasing, removing HTML, URLs, digits and punctuation, tokenization,
stopword removal and lemmatization. The full chain is demonstrated in
`pipeline/02_preprocessing.py`. The later stages each apply the parts they
need. VADER and NER deliberately get the **raw** text, because they need
negations, punctuation and capital letters.

**Q: Stemming or lemmatization? Why?**
Lemmatization (spaCy), because it returns real words using the part of speech
(`was → be`), while stemming just chops suffixes (`studies → studi`).

**Q: What are n-grams and what did you find?**
Sequences of n words. The top bigrams include "waste money", "waste time" and
"highly recommend". Phrases like these carry sentiment that single words miss.

**Q: Explain TF-IDF.**
Term frequency × inverse document frequency. A word is important to a document
if it is frequent there but rare in the corpus. Formula
`tf(t,d) × log(N / df(t))`. We use it for keywords (stage 04), as the input to
NMF (stage 08) and in the lexical search fallback.

**Q: TF-IDF vs RAKE?**
TF-IDF scores single terms using the whole corpus. RAKE finds multi-word key
phrases inside one document, using stopwords as phrase boundaries.

**Q: How does VADER work?**
It looks up each word in a lexicon of 7,506 rated entries, adjusts the scores
for negation, intensifiers, punctuation, capitals and "but", and normalises the
sum into a compound score from −1 to +1. ≥ 0.05 is positive, ≤ −0.05 negative.

**Q: Why is VADER's accuracy only 70%?**
It was built for general and social-media text. Many product complaints score
only weakly negative ("stopped working" −0.23) or neutral ("sent it back" 0.0),
so praise elsewhere in the review outvotes them: *"The screen is great and the
design is lovely but it stopped working after a week."* gets +0.38 (positive).
Some complaints even sound positive ("I really wanted to like it" scores +0.41).
VADER called 24,180 of 50,000 negative reviews positive.

**Q: Why a hybrid engine, and how are the two parts combined?**
The rule lexicon gives product-review phrases a fixed, heavy weight, so they are
not outvoted (the "screen is great… stopped working" review above becomes
Negative). If only negative (or
only positive) phrases fire, they decide. Otherwise
`final = 0.70 × VADER + 0.30 × rules`. A few decisive phrases ("waste of money")
always force negative. `app/sentiment.py → analyze()`.

**Q: Why 0.70 / 0.30, and why those phrase weights?**
They are hand-set heuristics, not learned values. VADER gets the larger share
because it handles negation and grammar. The rules act as a correction.
Stage 12 checks the result: 70.45% → 76.27%. A better approach would tune these
weights on a held-out validation set.

**Q: How do you know the hybrid is better?**
We measured it on all 100k labelled reviews (`pipeline/12_evaluate_sentiment.py`):
accuracy 70.45% → 76.27%, negative recall 0.485 → 0.615, negative F1
0.626 → 0.727.

**Q: Explain precision, recall and F1.**
Precision: of the predicted negatives, how many are right. Recall: of the true
negatives, how many we found. F1 is their harmonic mean. The hybrid mainly
improves negative recall.

**Q: Why can the final score be −4.2? Isn't sentiment between −1 and +1?**
VADER's compound is. Under a phrase override our final score is
compound + rule score, and the rule score grows with every matched phrase. The
sign decides the label; the size shows how strong the evidence was.

**Q: What is aspect-based sentiment analysis and how do you do it?**
Sentiment per product aspect instead of per review. We detect 28 predefined
aspect words with whole-word regex matching, attach the review's VADER label to
each aspect found, and count. `pipeline/07_aspect_sentiment.py`.

**Q: What is the weakness of your aspect method?**
The sentiment belongs to the whole review, not the aspect's sentence. "Great
battery, awful screen" gives both the same label. The aspect list is also
fixed, so new aspects are never discovered.

**Q: What is topic modelling and why NMF?**
Unsupervised discovery of themes. NMF factorises the TF-IDF matrix into
document-topic (W) and topic-term (H) matrices with non-negative values, which
gives readable, additive topics. It is fast on 100k documents.

**Q: NMF vs LDA?**
LDA is probabilistic (Dirichlet priors, usually fitted on word counts). NMF is
linear-algebraic and works on TF-IDF. Both give topics as weighted word lists.
NMF often gives cleaner topics on short texts.

**Q: Why 10 topics?**
It is a fixed parameter (`NUM_TOPICS = 10` in stage 08) chosen to be readable.
We checked by hand that the topics are coherent. A more rigorous choice would
compare topic-coherence scores for several values of k.

**Q: How did you name the topics?**
By hand, from each topic's top keywords (`app/config.py TOPIC_LABELS`). NMF only
outputs numbers. Manual labelling is standard practice.

**Q: Why is "don" a topic keyword?**
Cleaning removes the apostrophe in "don't", and the leftover one-letter "t" is
dropped by the vectorizer, which leaves "don".

**Q: What is TextRank?**
PageRank applied to sentences. Sentences are nodes, word overlap gives the edge
weights, and PageRank (damping 0.85) finds the most central sentences. They are
output in their original order. The live engine is the `sumy` library's
TextRank; `app/summarize.py → _textrank()` is a built-in version with the maths
written out (section 6.6 compares the two).

**Q: Extractive or abstractive? Why?**
Extractive. It can never invent facts, needs no training, and is fast.

**Q: Why are some summaries identical to the review?**
Reviews under 40 words (25 words in the live app), or with no more sentences
than requested, have nothing to condense. Of the 5,000 stored summaries,
2,063 are identical for these reasons (section 8.4).

**Q: What are embeddings and which model do you use?**
Vectors that encode meaning. all-MiniLM-L6-v2, a small 6-layer
sentence-transformer that outputs 384 numbers per text. Similar meanings land
close together.

**Q: What is cosine similarity and why not Euclidean distance?**
The cosine of the angle between two vectors. It measures direction (meaning)
and ignores length, so texts of different lengths with the same meaning still
match.

**Q: How does semantic search work?**
The query is embedded with the same model as the reviews, cosine similarity is
computed against all 5,000 review vectors, and the top k are returned.
`app/semantic.py → search()`.

**Q: Why only 5,000 reviews for search?**
The scripts don't record a reason. The likely one is speed: computing
Transformer embeddings (and TextRank summaries) is the slowest step, and 5,000
keeps it practical on a laptop. The app code works unchanged with more rows.

**Q: What happens if a library is missing?**
Each engine has a fallback: rules-only sentiment, built-in TextRank, lexical
search. The app keeps working and the left rail shows which engine answered.

**Q: Did you train a model?**
No supervised model is trained. VADER and our rules are lexicon-based. NMF is
fitted unsupervised on our corpus. MiniLM is pre-trained and used as-is. The
human labels are used only to evaluate.

**Q: Why FastAPI and a separate frontend?**
The NLP libraries are in Python, so the NLP runs on a Python server. The
browser only displays results. FastAPI validates the requests and
auto-documents the API (`/docs`).

**Q: What exactly happens when I click "Analyse"?**
`app.js` POSTs the text to `/api/sentiment`. `server.py` validates it and calls
`sentiment.analyze()`, which runs the rule lexicon, VADER, the combination and
the hard rules, and returns the label, scores, evidence spans and reasoning.
`app.js` highlights the evidence and lists the steps.
([Section 5.1](#51-how-a-click-becomes-a-result).)

**Q: Why load data lazily?**
The CSVs total about 260 MB. Loading each file the first time it is needed,
and keeping it in memory, makes start-up instant and keeps later requests
fast.

---

## 10. Limitations and possible improvements

| Limitation | Where | Improvement |
|---|---|---|
| Aspect sentiment uses the whole review's label | `pipeline/07` | Score only the sentence(s) mentioning the aspect, or use a dedicated ABSA model |
| Fixed list of 28 aspects | `pipeline/07` | Extract aspects automatically (noun phrases, or the n-gram / NER results) |
| Rule phrases match inside longer words and stack ("worst product" also counts "worst") | `app/sentiment.py`, `app/lexicon.py` | Match on word boundaries; count the longest match only |
| The rule lexicon has no negation handling ("not disappointed" counts as negative) | `app/sentiment.py` | Check for a negation word just before each match |
| Hybrid weights (0.70/0.30, ±0.40/0.50…) set by hand | `app/sentiment.py` | Tune them on a validation split |
| Sarcasm and polite complaints are missed ("I really wanted to like it" → Positive) | sentiment | A fine-tuned Transformer classifier (e.g. DistilBERT) |
| No neutral human labels, so neutral predictions can't be evaluated | dataset | Use a 5-star dataset and map 3 stars to neutral |
| Number of topics fixed at 10; two topics are opinion words rather than subjects | `pipeline/08` | Choose k by topic coherence; add sentiment words to the stopword list for NMF; try BERTopic |
| Search covers 5,000 of 100,000 reviews | `pipeline/10` | Embed all reviews and use a vector index (e.g. FAISS) |
| MiniLM reads at most 256 word-pieces | `pipeline/10` | Chunk long reviews and average the chunk embeddings |
| Stages 02–05 are exploratory and don't feed the app | `pipeline/` | Surface n-grams, keywords and entities in their own pages |
| English only | everywhere | Multilingual models (e.g. multilingual MiniLM) |

---

## 11. Five-minute demo script

Before the demo, install every engine (`pip install -r requirements.txt`) and
check that the left rail shows **retrieval: semantic** in green. Start with
`python run.py` and **keep that terminal window open** for the whole demo.
Closing it stops the server, and every page then shows "Cannot reach the
Python server". If that happens, run `python run.py` again and reload the page.

1. **Corpus (60 s).** "100,000 Amazon reviews, balanced 50/50." Point to the
    two polarity bars: "VADER leans positive and agrees with the human labels
    70.5% of the time." Point to the 11 pipeline stages. Open **How this
    works**. Scroll to the charts: "drawn from the full run by
    `app/plates.py`."
2. **Verdict (90 s).** Click **a complaint** → Analyse. Show the highlighted
    phrases and the numbered steps: "each verdict shows its arithmetic." Then
    type *The screen is great and the design is lovely but it stopped working
    after a week.* → Negative: "VADER alone says positive (+0.38) because the
    praise outvotes 'stopped working'; our phrase rule catches it." Mention
    the evaluation: 70.45% → 76.27%.
3. **Aspects (30 s).** Sort by **Most negative**, then click **Customer
    service**: "the only aspect with net-negative sentiment."
4. **Topics (45 s).** Click **Topic 10**: "NMF found this video-game topic
    from word co-occurrence alone: game, play, fun, graphics." Then Topic 4:
    "the complaint topic, 67% negative."
5. **Retrieval (60 s).** Search *stopped working after a few weeks*: "results
    match by meaning. Cosine similarity of 384-dimensional MiniLM
    embeddings." Then click **Find similar →** on the first result: "the same
    maths, but the query is a stored review vector, so no model runs."
6. **Condense (30 s).** **Load one from the corpus** → Condense: "TextRank
    keeps the most central sentences; nothing is rewritten."

---

## 12. Quick reference: where is it in the code?

| If they ask about… | Open |
|---|---|
| How the 100k dataset was built | `pipeline/01_prepare_dataset.py` |
| Preprocessing / lemmatization | `pipeline/02_preprocessing.py` |
| N-grams | `pipeline/03_ngram_analysis.py` |
| TF-IDF / RAKE | `pipeline/04_keyword_extraction.py` |
| NER | `pipeline/05_ner_analysis.py` |
| VADER over the corpus, the 70.45% figure | `pipeline/06_sentiment_analysis.py` |
| The 28 aspects | `pipeline/07_aspect_sentiment.py` → `ASPECTS` |
| NMF, number of topics | `pipeline/08_topic_extraction.py` → `NUM_TOPICS` |
| Topic names | `app/config.py` → `TOPIC_LABELS` |
| Stored summaries (5,000 reviews) | `pipeline/09_summarization.py` |
| Embedding model and output | `pipeline/10_embeddings_similarity.py`, `app/config.py` → `EMBEDDING_MODEL` |
| Evaluation, the 76.27% figure | `pipeline/12_evaluate_sentiment.py` |
| Live sentiment algorithm, weights, thresholds | `app/sentiment.py` (constants at the top, `analyze()`) |
| The phrase and keyword lists | `app/lexicon.py` |
| TextRank maths | `app/summarize.py` → `_textrank()`, `_sentence_similarity()` (built-in version; the live engine is sumy's) |
| Cosine similarity | `app/semantic.py` → `cosine_similarity()` |
| Search and neighbours | `app/semantic.py` → `search()`, `neighbours()` |
| All API endpoints | `app/server.py` (map in its header) |
| Where data files are loaded | `app/data.py` |
| The page layout and "How this works" text | `web/index.html` |
| What each page does in the browser | `web/assets/app.js` (one section per page) |
| Tests | `tests/` (`python -m pytest`) |
