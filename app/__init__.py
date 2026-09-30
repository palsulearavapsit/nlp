"""The Review Room - the web application package.

This package is the ONLINE half of the project. The OFFLINE half is the
pipeline/ folder, which ran the heavy NLP once and saved the results as CSV
files. This package loads those CSVs, runs a few NLP engines live on text you
type, and serves everything to the browser.

    config.py     where every file lives, and tunable limits
    data.py       loads the CSV / embedding files once and caches them
    lexicon.py    word and phrase lists for product-review sentiment
    sentiment.py  LIVE sentiment engine: VADER + lexicon rules, with evidence
    summarize.py  LIVE extractive summarizer: TextRank
    semantic.py   LIVE semantic search and nearest-neighbour similarity
    server.py     the HTTP API (FastAPI) the browser talks to
    plates.py     redraws the chart images shown on the Corpus page

Start it with:  python run.py
"""

__version__ = "2.0.0"
