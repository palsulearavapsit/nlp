"""Render the analytical plates in The Review Room's palette.

The originals were stock matplotlib defaults, which fought the interface they
sit in. These redraw the same six figures from the same source CSVs using the
design system's ink, newsprint and signal colours.

    python -m app.plates
"""

import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.ticker import FuncFormatter  # noqa: E402

from app import config, data  # noqa: E402

# Design tokens, mirrored from web/assets/styles.css.
PAPER = "#F0EBE1"
PAPER_2 = "#E7E1D4"
INK = "#14130F"
INK_3 = "#6A6559"
SIGNAL = "#D8380F"
POSITIVE = "#1E6B49"
NEGATIVE = "#B33417"
NEUTRAL = "#6B7688"

SERIF = ["Georgia", "Cambria", "Times New Roman", "serif"]
MONO = ["Consolas", "DejaVu Sans Mono", "monospace"]

FIGSIZE = (10, 5.6)
DPI = 150

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from
further get got had has have having he her here hers him his how i if in into is it its
just me more most my no nor not of off on once only or other our out over own same she
so some such than that the their them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would
you your s t don ve ll re m
""".split())

WORD_RE = re.compile(r"[a-z']{3,}")


def _canvas(title, subtitle):
    figure, axes = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    figure.patch.set_facecolor(PAPER)
    axes.set_facecolor(PAPER)

    figure.text(0.055, 0.955, title, fontfamily=SERIF, fontsize=19,
                color=INK, va="top")
    figure.text(0.055, 0.888, subtitle.upper(), fontfamily=MONO, fontsize=8.5,
                color=INK_3, va="top", linespacing=1.4)
    figure.text(0.945, 0.955, "THE REVIEW ROOM", fontfamily=MONO, fontsize=7.5,
                color=SIGNAL, va="top", ha="right")

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(INK)
        axes.spines[side].set_linewidth(1)

    axes.tick_params(colors=INK_3, labelsize=8.5, length=4, width=1, color=INK)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontfamily(MONO)

    axes.grid(axis="y", color=INK, alpha=0.10, linewidth=0.8)
    axes.set_axisbelow(True)
    figure.subplots_adjust(left=0.075, right=0.955, top=0.80, bottom=0.16)
    return figure, axes


def _thousands(axes, *which):
    """Group the tick labels of the given numeric axes with commas."""
    formatter = FuncFormatter(lambda value, _: f"{int(value):,}")
    for name in which:
        getattr(axes, f"{name}axis").set_major_formatter(formatter)


def _save(figure, filename):
    path = os.path.join(config.VISUALS_DIR, filename)
    figure.savefig(path, facecolor=PAPER, dpi=DPI)
    plt.close(figure)
    print(f"  rendered {filename}")


def _label_bars(axes, bars, values, colour=INK):
    for bar, value in zip(bars, values):
        axes.annotate(f"{int(value):,}",
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 4), textcoords="offset points",
                      ha="center", fontfamily=MONO, fontsize=7.5, color=colour)


def sentiment_distribution():
    labels = data.sentiment_labels()
    counts = labels["vader_sentiment"].astype(str).str.lower().value_counts()
    order = ["positive", "neutral", "negative"]
    values = [int(counts.get(key, 0)) for key in order]

    figure, axes = _canvas("Sentiment Distribution",
                           f"predicted polarity across {len(labels):,} reviews")
    bars = axes.bar(order, values,
                    color=[POSITIVE, NEUTRAL, NEGATIVE], edgecolor=INK, linewidth=1)
    _label_bars(axes, bars, values)
    axes.set_ylabel("Reviews", fontfamily=MONO, fontsize=8.5, color=INK_3)
    axes.set_ylim(0, max(values) * 1.16)
    _thousands(axes, "y")
    _save(figure, "sentiment_distribution.png")


def topic_distribution():
    frame = data.topics()
    counts = frame["topic"].value_counts().sort_index()
    values = [int(v) for v in counts.values]
    peak = max(values)

    figure, axes = _canvas("Topic Distribution",
                           f"{len(counts)} latent clusters · {len(frame):,} reviews")
    bars = axes.bar([str(k) for k in counts.index], values,
                    color=[SIGNAL if v == peak else INK for v in values],
                    edgecolor=INK, linewidth=1)
    _label_bars(axes, bars, values)
    axes.set_xlabel("Topic", fontfamily=MONO, fontsize=8.5, color=INK_3)
    axes.set_ylabel("Reviews", fontfamily=MONO, fontsize=8.5, color=INK_3)
    axes.set_ylim(0, peak * 1.16)
    _thousands(axes, "y")
    _save(figure, "topic_distribution.png")


def aspect_frequency():
    frame = data.aspect_summary().copy()
    frame["total"] = frame[["negative", "neutral", "positive"]].sum(axis=1)
    frame = frame.sort_values("total", ascending=True)

    figure, axes = _canvas("Aspect Frequency",
                           f"{len(frame)} product facets by mention volume")
    bars = axes.barh(frame["aspect"], frame["total"],
                     color=INK, edgecolor=INK, linewidth=1, height=0.72)
    bars[-1].set_color(SIGNAL)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=INK, alpha=0.10, linewidth=0.8)
    axes.set_xlabel("Mentions", fontfamily=MONO, fontsize=8.5, color=INK_3)
    axes.set_xlim(0, frame["total"].max() * 1.10)
    _thousands(axes, "x")
    figure.set_size_inches(10, 8.2)
    figure.subplots_adjust(left=0.17, top=0.86, bottom=0.09)
    _save(figure, "aspect_frequency.png")


def aspect_sentiment():
    frame = data.aspect_summary().copy()
    frame["total"] = frame[["negative", "neutral", "positive"]].sum(axis=1)
    frame = frame.sort_values("total", ascending=False).head(16)
    frame = frame.iloc[::-1]

    figure, axes = _canvas("Aspect Sentiment",
                           "polarity composition within the 16 loudest aspects")
    left = [0.0] * len(frame)
    for column, colour in (("positive", POSITIVE), ("neutral", NEUTRAL), ("negative", NEGATIVE)):
        share = frame[column] / frame["total"] * 100
        axes.barh(frame["aspect"], share, left=left, color=colour,
                  edgecolor=PAPER, linewidth=1, height=0.74, label=column)
        left = [a + b for a, b in zip(left, share)]

    axes.grid(visible=False)
    axes.set_xlim(0, 100)
    axes.set_xlabel("Share of mentions (%)", fontfamily=MONO, fontsize=8.5, color=INK_3)
    legend = axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3,
                         frameon=False, prop={"family": MONO, "size": 8.5})
    for text in legend.get_texts():
        text.set_color(INK_3)
    figure.set_size_inches(10, 7)
    figure.subplots_adjust(left=0.17, top=0.84, bottom=0.16)
    _save(figure, "aspect_sentiment.png")


def review_length_distribution():
    frame = data.topics()
    lengths = frame["review"].astype(str).str.split().str.len()
    trimmed = lengths[lengths <= lengths.quantile(0.99)]

    figure, axes = _canvas("Review Length",
                           f"words per review · median {int(lengths.median())} · "
                           f"99th percentile trimmed")
    axes.hist(trimmed, bins=60, color=INK, edgecolor=PAPER, linewidth=0.4)
    axes.axvline(lengths.median(), color=SIGNAL, linewidth=1.6)
    axes.annotate(f"median {int(lengths.median())}",
                  (lengths.median(), axes.get_ylim()[1] * 0.92),
                  xytext=(8, 0), textcoords="offset points",
                  fontfamily=MONO, fontsize=8, color=SIGNAL)
    axes.set_xlabel("Words", fontfamily=MONO, fontsize=8.5, color=INK_3)
    axes.set_ylabel("Reviews", fontfamily=MONO, fontsize=8.5, color=INK_3)
    _thousands(axes, "x", "y")
    _save(figure, "review_length_distribution.png")


def top_words():
    frame = data.topics()
    counter = Counter()
    for text in frame["review"].astype(str).head(40000):
        counter.update(w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS)

    common = counter.most_common(22)[::-1]
    words = [word for word, _ in common]
    values = [count for _, count in common]

    figure, axes = _canvas("Top Words",
                           "highest-frequency terms after stopword removal")
    axes.barh(words, values, color=INK, edgecolor=INK, linewidth=1, height=0.72)
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=INK, alpha=0.10, linewidth=0.8)
    axes.set_xlabel("Occurrences", fontfamily=MONO, fontsize=8.5, color=INK_3)
    _thousands(axes, "x")
    figure.set_size_inches(10, 8.2)
    figure.subplots_adjust(left=0.14, top=0.86, bottom=0.09)
    _save(figure, "top_words.png")


PLATES = (
    sentiment_distribution,
    topic_distribution,
    aspect_frequency,
    aspect_sentiment,
    review_length_distribution,
    top_words,
)


def main():
    os.makedirs(config.VISUALS_DIR, exist_ok=True)
    print("Rendering plates…")
    for plate in PLATES:
        plate()
    print("Done.")


if __name__ == "__main__":
    main()
