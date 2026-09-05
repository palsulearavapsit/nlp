import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Review NLP Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 600;
        margin-top: 10px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }

    .positive-box {
        background-color: rgba(34, 197, 94, 0.15);
        border-left: 5px solid #22c55e;
    }

    .negative-box {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #ef4444;
    }

    .neutral-box {
        background-color: rgba(59, 130, 246, 0.15);
        border-left: 5px solid #3b82f6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SENTIMENT_FILE = os.path.join(
    BASE_DIR,
    "amazon_reviews_sentiment.csv"
)

ASPECT_FILE = os.path.join(
    BASE_DIR,
    "amazon_review_aspects.csv"
)

ASPECT_SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "aspect_sentiment_summary.csv"
)

TOPIC_FILE = os.path.join(
    BASE_DIR,
    "amazon_reviews_topics.csv"
)

SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "amazon_reviews_summarized.csv"
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "review_embeddings.csv"
)

VISUALS_DIR = os.path.join(
    BASE_DIR,
    "visuals"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)

    return None


@st.cache_data
def load_embeddings(path):

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # Keep only numeric columns
    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    # SentenceTransformer MiniLM embeddings = 384 dimensions
    if numeric_df.shape[1] >= 384:

        numeric_df = numeric_df.iloc[:, -384:]

        return numeric_df.values.astype(
            "float32"
        )

    return numeric_df.values.astype(
        "float32"
    )


@st.cache_resource
def load_sentence_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vader():

    return SentimentIntensityAnalyzer()


sentiment_df = load_csv(
    SENTIMENT_FILE
)

aspect_df = load_csv(
    ASPECT_FILE
)

aspect_summary_df = load_csv(
    ASPECT_SUMMARY_FILE
)

topic_df = load_csv(
    TOPIC_FILE
)

summary_df = load_csv(
    SUMMARY_FILE
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def find_column(df, possible_names):

    if df is None:
        return None

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        for name in possible_names:

            if name.lower() in column_lower:
                return column

    return None


def get_review_column(df):

    if df is None:
        return None

    possible = [
        "review",
        "review_text",
        "text",
        "combined_text",
        "content"
    ]

    column = find_column(
        df,
        possible
    )

    if column:
        return column

    object_columns = df.select_dtypes(
        include="object"
    ).columns

    if len(object_columns) > 0:
        return object_columns[0]

    return None


def cosine_similarity(
    query_embedding,
    embeddings
):

    query_embedding = np.asarray(
        query_embedding
    )

    embeddings = np.asarray(
        embeddings
    )

    query_norm = np.linalg.norm(
        query_embedding
    )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    return np.dot(
        embeddings,
        query_embedding
    ) / (
        embedding_norms * query_norm + 1e-10
    )


# ============================================================
# IMPROVED NLP SENTIMENT ANALYSIS
# ============================================================

NEGATIVE_PHRASES = [

    "waste of money",
    "waste money",
    "waste of time",
    "waste time",
    "not worth",
    "not worth the money",
    "do not buy",
    "do not purchase",
    "don't buy",
    "dont buy",
    "don't purchase",
    "dont purchase",
    "should not buy",
    "should not be bought",
    "shouldn't buy",
    "shouldn't be bought",
    "avoid this",
    "avoid this product",
    "stay away",
    "never buy",
    "never purchase",
    "poor quality",
    "terrible quality",
    "bad quality",
    "low quality",
    "very slow",
    "too slow",
    "extremely slow",
    "does not work",
    "doesn't work",
    "doesnt work",
    "not working",
    "stopped working",
    "barely works",
    "hardly works",
    "broken",
    "broke",
    "defective",
    "useless",
    "disappointed",
    "disappointing",
    "disappointment",
    "horrible",
    "terrible",
    "awful",
    "worst",
    "worst product",
    "worst purchase",
    "bad product",
    "bad purchase",
    "poor product",
    "poor performance",
    "poor service",
    "poor design",
    "poor sound",
    "poor battery",
    "poor screen",
    "poor material",
    "too expensive",
    "overpriced",
    "not recommended",
    "would not recommend",
    "wouldn't recommend",
    "do not recommend",
    "not good",
    "not great",
    "not happy",
    "not satisfied",
    "very disappointing",
    "highly disappointing",
    "return this",
    "returned this",
    "sent it back",
    "send it back"
]


POSITIVE_PHRASES = [

    "highly recommend",
    "strongly recommend",
    "would recommend",
    "definitely recommend",
    "must buy",
    "great product",
    "excellent product",
    "excellent quality",
    "great quality",
    "good quality",
    "amazing quality",
    "fantastic quality",
    "works perfectly",
    "works great",
    "works well",
    "works perfectly well",
    "very good",
    "very great",
    "really good",
    "really great",
    "extremely good",
    "extremely great",
    "very happy",
    "very satisfied",
    "really happy",
    "really satisfied",
    "love this",
    "love the product",
    "love it",
    "excellent",
    "amazing",
    "fantastic",
    "awesome",
    "wonderful",
    "perfect",
    "best product",
    "best purchase",
    "worth the money",
    "worth every penny",
    "good value",
    "great value",
    "excellent value",
    "fast delivery",
    "quick delivery",
    "easy to use",
    "easy to install",
    "easy to setup",
    "high quality"
]


NEGATIVE_WORDS = [
    "bad",
    "terrible",
    "horrible",
    "awful",
    "worst",
    "poor",
    "broken",
    "useless",
    "defective",
    "disappointed",
    "disappointing",
    "slow",
    "expensive",
    "overpriced",
    "hate",
    "refund",
    "return",
    "problem",
    "problems",
    "failure",
    "fail",
    "failed",
    "damage",
    "damaged",
    "annoying",
    "waste"
]


POSITIVE_WORDS = [
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "fantastic",
    "wonderful",
    "perfect",
    "love",
    "best",
    "happy",
    "satisfied",
    "recommend",
    "recommended",
    "easy",
    "quality",
    "fast",
    "useful",
    "value"
]


def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def phrase_score(
    text
):

    text = normalize_text(
        text
    )

    negative_hits = []
    positive_hits = []

    for phrase in NEGATIVE_PHRASES:

        if phrase in text:
            negative_hits.append(
                phrase
            )

    for phrase in POSITIVE_PHRASES:

        if phrase in text:
            positive_hits.append(
                phrase
            )

    # Individual words
    words = set(
        re.findall(
            r"\b[a-z]+\b",
            text
        )
    )

    negative_word_hits = [
        word
        for word in NEGATIVE_WORDS
        if word in words
    ]

    positive_word_hits = [
        word
        for word in POSITIVE_WORDS
        if word in words
    ]

    score = (
        len(positive_hits) * 0.40
        + len(positive_word_hits) * 0.08
        - len(negative_hits) * 0.50
        - len(negative_word_hits) * 0.10
    )

    return (
        score,
        negative_hits,
        positive_hits,
        negative_word_hits,
        positive_word_hits
    )


def analyze_sentiment(text):

    analyzer = load_vader()

    vader_scores = analyzer.polarity_scores(
        text
    )

    compound = vader_scores[
        "compound"
    ]

    (
        rule_score,
        negative_phrases,
        positive_phrases,
        negative_words,
        positive_words
    ) = phrase_score(
        text
    )

    # --------------------------------------------------------
    # Strong explicit product-review phrases
    # take priority over VADER when appropriate.
    # --------------------------------------------------------

    if negative_phrases and not positive_phrases:

        sentiment = "Negative"

        final_score = min(
            -0.30,
            compound + rule_score
        )

    elif positive_phrases and not negative_phrases:

        sentiment = "Positive"

        final_score = max(
            0.30,
            compound + rule_score
        )

    else:

        final_score = (
            compound * 0.70
            + rule_score * 0.30
        )

        if final_score >= 0.05:

            sentiment = "Positive"

        elif final_score <= -0.05:

            sentiment = "Negative"

        else:

            sentiment = "Neutral"

    # Product-review special cases
    text_lower = normalize_text(text)

    if (
        "very slow" in text_lower
        or "too slow" in text_lower
        or "extremely slow" in text_lower
    ):

        sentiment = "Negative"
        final_score = -0.60

    if (
        "should not be bought" in text_lower
        or "should not buy" in text_lower
        or "do not buy" in text_lower
        or "don't buy" in text_lower
    ):

        sentiment = "Negative"
        final_score = -0.80

    if (
        "waste of money" in text_lower
        or "waste money" in text_lower
    ):

        sentiment = "Negative"
        final_score = -0.80

    return {
        "sentiment": sentiment,
        "compound": compound,
        "final_score": final_score,
        "positive": vader_scores["pos"],
        "negative": vader_scores["neg"],
        "neutral": vader_scores["neu"],
        "negative_phrases": negative_phrases,
        "positive_phrases": positive_phrases,
        "negative_words": negative_words,
        "positive_words": positive_words
    }


# ============================================================
# TEXT SUMMARIZATION
# ============================================================

def summarize_text(
    text,
    sentence_count=3
):

    text = str(text)

    if len(text.split()) < 25:
        return text

    try:

        parser = PlaintextParser.from_string(
            text,
            Tokenizer("english")
        )

        summarizer = TextRankSummarizer()

        sentences = summarizer(
            parser.document,
            sentences_count=sentence_count
        )

        result = " ".join(
            str(sentence)
            for sentence in sentences
        )

        if result:
            return result

        return text

    except Exception:

        return text


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 NLP Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Overview",
        "😊 Sentiment Analysis",
        "🔎 Aspect-Based Sentiment",
        "📚 Topic Extraction",
        "🔍 Semantic Search",
        "🔗 Similar Reviews",
        "📝 Text Summarization",
        "📊 NLP Visualizations"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Amazon Review NLP Intelligence System\n\n"
    "NLP-only analytical pipeline using "
    "VADER, spaCy, TextRank and "
    "Sentence Transformers."
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        '🧠 Amazon Review NLP Intelligence System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An NLP-based customer review intelligence platform'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_reviews = 0
    positive_reviews = 0
    negative_reviews = 0
    neutral_reviews = 0

    if sentiment_df is not None:

        total_reviews = len(
            sentiment_df
        )

        sentiment_column = find_column(
            sentiment_df,
            ["sentiment"]
        )

        if sentiment_column:

            values = (
                sentiment_df[
                    sentiment_column
                ]
                .astype(str)
                .str.lower()
            )

            positive_reviews = (
                values == "positive"
            ).sum()

            negative_reviews = (
                values == "negative"
            ).sum()

            neutral_reviews = (
                values == "neutral"
            ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        f"{total_reviews:,}"
    )

    col2.metric(
        "Positive",
        f"{positive_reviews:,}"
    )

    col3.metric(
        "Negative",
        f"{negative_reviews:,}"
    )

    col4.metric(
        "Neutral",
        f"{neutral_reviews:,}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.subheader(
        "NLP Processing Pipeline"
    )

    pipeline = [
        "Amazon Reviews",
        "Text Preprocessing",
        "N-Gram Analysis",
        "Keyword Extraction",
        "Named Entity Recognition",
        "Sentiment Analysis",
        "Aspect-Based Sentiment",
        "Topic Extraction",
        "Text Summarization",
        "Semantic Similarity",
        "Semantic Search",
        "NLP Analytics",
        "Interactive Dashboard"
    ]

    for i, step in enumerate(
        pipeline
    ):

        st.markdown(
            f"**{i + 1}. {step}**"
        )

        if i < len(pipeline) - 1:
            st.write("↓")

    st.markdown("---")

    # --------------------------------------------------------
    # Main Visuals
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    sentiment_image = os.path.join(
        VISUALS_DIR,
        "sentiment_distribution.png"
    )

    topic_image = os.path.join(
        VISUALS_DIR,
        "topic_distribution.png"
    )

    with col1:

        if os.path.exists(
            sentiment_image
        ):

            st.image(
                sentiment_image,
                caption="Sentiment Distribution",
                use_container_width=True
            )

    with col2:

        if os.path.exists(
            topic_image
        ):

            st.image(
                topic_image,
                caption="Topic Distribution",
                use_container_width=True
            )


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

elif page == "😊 Sentiment Analysis":

    st.header(
        "😊 Sentiment Analysis"
    )

    st.write(
        "Analyze customer reviews using a hybrid "
        "VADER + product-review NLP rule system."
    )

    review = st.text_area(
        "Enter Customer Review",
        height=180,
        placeholder=(
            "Example: The massager is very slow "
            "and should not be bought at any cost."
        )
    )

    if st.button(
        "🔍 Analyze Sentiment",
        use_container_width=False
    ):

        if not review.strip():

            st.warning(
                "Please enter a review."
            )

        else:

            result = analyze_sentiment(
                review
            )

            sentiment = result[
                "sentiment"
            ]

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            if sentiment == "Positive":

                st.success(
                    f"### 😊 Predicted Sentiment: {sentiment}"
                )

            elif sentiment == "Negative":

                st.error(
                    f"### 😞 Predicted Sentiment: {sentiment}"
                )

            else:

                st.info(
                    f"### 😐 Predicted Sentiment: {sentiment}"
                )

            st.markdown("---")

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "VADER Positive",
                f"{result['positive']:.3f}"
            )

            col2.metric(
                "VADER Negative",
                f"{result['negative']:.3f}"
            )

            col3.metric(
                "VADER Neutral",
                f"{result['neutral']:.3f}"
            )

            col4.metric(
                "VADER Compound",
                f"{result['compound']:.3f}"
            )

            st.markdown("---")

            # ------------------------------------------------
            # NLP Evidence
            # ------------------------------------------------

            st.subheader(
                "NLP Evidence"
            )

            if result[
                "negative_phrases"
            ]:

                st.error(
                    "Negative phrases detected: "
                    + ", ".join(
                        result[
                            "negative_phrases"
                        ]
                    )
                )

            if result[
                "positive_phrases"
            ]:

                st.success(
                    "Positive phrases detected: "
                    + ", ".join(
                        result[
                            "positive_phrases"
                        ]
                    )
                )

            if result[
                "negative_words"
            ]:

                st.write(
                    "**Negative keywords:** "
                    + ", ".join(
                        result[
                            "negative_words"
                        ]
                    )
                )

            if result[
                "positive_words"
            ]:

                st.write(
                    "**Positive keywords:** "
                    + ", ".join(
                        result[
                            "positive_words"
                        ]
                    )
                )

            st.write(
                f"**Final NLP sentiment score:** "
                f"{result['final_score']:.3f}"
            )

    st.markdown("---")

    sentiment_image = os.path.join(
        VISUALS_DIR,
        "sentiment_distribution.png"
    )

    if os.path.exists(
        sentiment_image
    ):

        st.image(
            sentiment_image,
            caption="Dataset Sentiment Distribution",
            use_container_width=True
        )


# ============================================================
# ASPECT SENTIMENT
# ============================================================

elif page == "🔎 Aspect-Based Sentiment":

    st.header(
        "🔎 Aspect-Based Sentiment Analysis"
    )

    st.write(
        "Explore sentiment associated with individual "
        "product-review aspects."
    )

    if aspect_summary_df is None:

        st.error(
            "aspect_sentiment_summary.csv not found."
        )

    else:

        st.dataframe(
            aspect_summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        aspect_column = find_column(
            aspect_summary_df,
            ["aspect"]
        )

        if aspect_column:

            selected_aspect = st.selectbox(
                "Select an Aspect",
                sorted(
                    aspect_summary_df[
                        aspect_column
                    ]
                    .dropna()
                    .unique()
                )
            )

            selected_row = aspect_summary_df[
                aspect_summary_df[
                    aspect_column
                ]
                == selected_aspect
            ]

            st.subheader(
                f"Analysis: {selected_aspect}"
            )

            st.dataframe(
                selected_row,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        image1 = os.path.join(
            VISUALS_DIR,
            "aspect_frequency.png"
        )

        image2 = os.path.join(
            VISUALS_DIR,
            "aspect_sentiment.png"
        )

        with col1:

            if os.path.exists(image1):

                st.image(
                    image1,
                    caption="Aspect Frequency",
                    use_container_width=True
                )

        with col2:

            if os.path.exists(image2):

                st.image(
                    image2,
                    caption="Aspect Sentiment",
                    use_container_width=True
                )


# ============================================================
# TOPIC EXTRACTION
# ============================================================

elif page == "📚 Topic Extraction":

    st.header(
        "📚 Topic Extraction"
    )

    st.write(
        "Explore the major themes identified in the "
        "Amazon review collection."
    )

    if topic_df is None:

        st.error(
            "amazon_reviews_topics.csv not found."
        )

    else:

        topic_column = find_column(
            topic_df,
            ["topic"]
        )

        review_column = get_review_column(
            topic_df
        )

        if topic_column:

            topics = sorted(
                topic_df[
                    topic_column
                ]
                .dropna()
                .unique()
            )

            selected_topic = st.selectbox(
                "Select Topic",
                topics
            )

            topic_reviews = topic_df[
                topic_df[
                    topic_column
                ]
                == selected_topic
            ]

            st.metric(
                "Reviews in Selected Topic",
                f"{len(topic_reviews):,}"
            )

            if review_column:

                st.subheader(
                    "Sample Reviews"
                )

                for i, text in enumerate(
                    topic_reviews[
                        review_column
                    ]
                    .head(10)
                ):

                    with st.expander(
                        f"Review {i + 1}"
                    ):

                        st.write(
                            str(text)
                        )

        st.markdown("---")

        topic_image = os.path.join(
            VISUALS_DIR,
            "topic_distribution.png"
        )

        if os.path.exists(
            topic_image
        ):

            st.image(
                topic_image,
                caption="Topic Distribution",
                use_container_width=True
            )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

elif page == "🔍 Semantic Search":

    st.header(
        "🔍 Semantic Search"
    )

    st.write(
        "Search reviews by semantic meaning rather than "
        "exact keyword matching."
    )

    query = st.text_input(
        "Enter Search Query",
        placeholder=(
            "customers complaining about poor product quality"
        )
    )

    top_k = st.slider(
        "Number of Results",
        min_value=3,
        max_value=15,
        value=5
    )

    if st.button(
        "🔍 Search Reviews"
    ):

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            if summary_df is None:

                st.error(
                    "amazon_reviews_summarized.csv not found."
                )

            else:

                with st.spinner(
                    "Searching semantically..."
                ):

                    embeddings = load_embeddings(
                        EMBEDDINGS_FILE
                    )

                    model = load_sentence_model()

                    if embeddings is None:

                        st.error(
                            "review_embeddings.csv not found."
                        )

                    else:

                        query_embedding = model.encode(
                            query,
                            convert_to_numpy=True
                        )

                        scores = cosine_similarity(
                            query_embedding,
                            embeddings
                        )

                        indices = np.argsort(
                            scores
                        )[::-1][:top_k]

                        review_column = get_review_column(
                            summary_df
                        )

                        st.success(
                            f"Found {len(indices)} "
                            "semantically similar reviews."
                        )

                        for rank, idx in enumerate(
                            indices,
                            start=1
                        ):

                            st.subheader(
                                f"Result {rank}"
                            )

                            st.write(
                                f"Semantic Similarity: "
                                f"**{scores[idx]:.4f}**"
                            )

                            if (
                                review_column
                                and idx < len(summary_df)
                            ):

                                st.write(
                                    str(
                                        summary_df.iloc[
                                            idx
                                        ][review_column]
                                    )
                                )

                            st.markdown("---")


# ============================================================
# SIMILAR REVIEWS
# ============================================================

elif page == "🔗 Similar Reviews":

    st.header(
        "🔗 Similar Review Finder"
    )

    st.write(
        "Select a review and find other reviews "
        "with similar semantic meaning."
    )

    if summary_df is None:

        st.error(
            "amazon_reviews_summarized.csv not found."
        )

    else:

        embeddings = load_embeddings(
            EMBEDDINGS_FILE
        )

        review_column = get_review_column(
            summary_df
        )

        if embeddings is None:

            st.error(
                "review_embeddings.csv not found."
            )

        elif review_column is None:

            st.error(
                "Review text column could not be identified."
            )

        else:

            available = min(
                len(summary_df),
                len(embeddings)
            )

            review_index = st.number_input(
                "Review Index",
                min_value=0,
                max_value=available - 1,
                value=0,
                step=1
            )

            selected_review = str(
                summary_df.iloc[
                    int(review_index)
                ][review_column]
            )

            st.subheader(
                "Selected Review"
            )

            st.info(
                selected_review
            )

            top_k = st.slider(
                "Number of Similar Reviews",
                min_value=3,
                max_value=15,
                value=5
            )

            if st.button(
                "🔗 Find Similar Reviews"
            ):

                selected_embedding = embeddings[
                    int(review_index)
                ]

                scores = cosine_similarity(
                    selected_embedding,
                    embeddings
                )

                indices = np.argsort(
                    scores
                )[::-1]

                indices = [
                    idx
                    for idx in indices
                    if idx != int(review_index)
                ]

                indices = indices[:top_k]

                for rank, idx in enumerate(
                    indices,
                    start=1
                ):

                    st.subheader(
                        f"Similar Review {rank}"
                    )

                    st.write(
                        f"Similarity: "
                        f"**{scores[idx]:.4f}**"
                    )

                    st.write(
                        str(
                            summary_df.iloc[
                                idx
                            ][review_column]
                        )
                    )

                    st.markdown("---")


# ============================================================
# TEXT SUMMARIZATION
# ============================================================

elif page == "📝 Text Summarization":

    st.header(
        "📝 Text Summarization"
    )

    st.write(
        "Generate a concise summary of a customer review "
        "using the TextRank NLP algorithm."
    )

    text = st.text_area(
        "Enter Review",
        height=300,
        placeholder=(
            "Paste a long customer review here..."
        )
    )

    sentence_count = st.slider(
        "Number of Summary Sentences",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button(
        "📝 Generate Summary"
    ):

        if not text.strip():

            st.warning(
                "Please enter text."
            )

        else:

            with st.spinner(
                "Generating summary..."
            ):

                summary = summarize_text(
                    text,
                    sentence_count
                )

            st.subheader(
                "Original Review"
            )

            st.write(
                text
            )

            st.markdown("---")

            st.subheader(
                "Generated Summary"
            )

            st.success(
                summary
            )


# ============================================================
# NLP VISUALIZATIONS
# ============================================================

elif page == "📊 NLP Visualizations":

    st.header(
        "📊 NLP Analytics & Visualizations"
    )

    visualizations = [

        (
            "Sentiment Distribution",
            "sentiment_distribution.png"
        ),

        (
            "Topic Distribution",
            "topic_distribution.png"
        ),

        (
            "Aspect Frequency",
            "aspect_frequency.png"
        ),

        (
            "Aspect Sentiment",
            "aspect_sentiment.png"
        ),

        (
            "Review Length Distribution",
            "review_length_distribution.png"
        ),

        (
            "Top Words",
            "top_words.png"
        )
    ]

    for title, filename in visualizations:

        st.subheader(
            title
        )

        image_path = os.path.join(
            VISUALS_DIR,
            filename
        )

        if os.path.exists(
            image_path
        ):

            st.image(
                image_path,
                use_container_width=True
            )

        else:

            st.warning(
                f"{filename} not found."
            )

        st.markdown("---")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Amazon Review NLP Intelligence System | "
    "NLP-based customer review analytics"
)