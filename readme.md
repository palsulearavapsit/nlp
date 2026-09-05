# 🧠 Amazon Review NLP Intelligence System

An NLP-based customer review intelligence system that analyzes Amazon customer reviews using multiple Natural Language Processing techniques.

The system transforms unstructured customer reviews into meaningful insights such as sentiment, important aspects, keywords, topics, summaries, semantic similarity and meaning-based search.

---

## 📌 Introduction

Online customer reviews contain valuable information about product quality, customer satisfaction, complaints and preferences. However, manually analyzing thousands of reviews is time-consuming and difficult.

The **Amazon Review NLP Intelligence System** uses Natural Language Processing to automatically analyze customer reviews and extract useful information.

The project combines multiple NLP applications into a single interactive dashboard.

### Main NLP Applications

- Text Preprocessing
- N-Gram Analysis
- Keyword Extraction
- Named Entity Recognition
- Sentiment Analysis
- Aspect-Based Sentiment Analysis
- Topic Extraction
- Text Summarization
- Semantic Similarity
- Semantic Search
- NLP Analytics and Visualization

---

# 🎯 Objectives

### 1. Analyze Customer Sentiment
Identify whether customer opinions expressed in reviews are positive, negative or neutral.

### 2. Extract Meaningful Information
Extract important keywords, phrases, entities and product-related aspects from unstructured review text.

### 3. Discover Customer Feedback Patterns
Identify major topics, summarize long reviews and discover semantically similar reviews.

### 4. Build an Interactive NLP Intelligence System
Integrate the different NLP applications into an interactive dashboard for customer review exploration and analysis.

---

# ⭐ Key Features

### 1. Sentiment Intelligence
Analyzes customer reviews and determines their sentiment using NLP-based sentiment analysis.

### 2. Aspect & Topic Intelligence
Identifies important product aspects and major themes discussed by customers.

### 3. Semantic Review Search
Allows users to search customer reviews based on meaning rather than only exact keyword matching.

### 4. Review Summarization & Similarity
Generates concise summaries of reviews and finds semantically similar reviews.

---

# 🔄 NLP Pipeline

```text
                         AMAZON REVIEWS
                               │
                               ▼
                    ┌─────────────────────┐
                    │     DATASET PREP    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  TEXT PREPROCESSING │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        N-GRAM ANALYSIS   KEYWORD EXTRACTION    NER
             │                 │
             │            ┌────┴────┐
             │            ▼         ▼
             │          TF-IDF     RAKE
             │
             └─────────────────┬─────────────────
                               │
                               ▼
                    ┌─────────────────────┐
                    │  SENTIMENT ANALYSIS │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ ASPECT-BASED        │
                    │ SENTIMENT ANALYSIS  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   TOPIC EXTRACTION  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  TEXT SUMMARIZATION │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SEMANTIC SIMILARITY │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SEMANTIC SEARCH   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ NLP ANALYTICS       │
                    │ & VISUALIZATION     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     DASHBOARD       │
                    └─────────────────────┘
```

---

# 🔍 Pipeline Explanation

## 1. Dataset Preparation

A balanced working dataset containing **100,000 Amazon reviews** is used for the project.

The dataset contains:

- 50,000 positive reviews
- 50,000 negative reviews

The original large dataset was used during development to create the working dataset. The final repository contains the prepared working dataset rather than the original large train/test files.

---

## 2. Text Preprocessing

The review text is cleaned and normalized before applying NLP techniques.

Operations include:

- Combining review title and review text
- Removing HTML
- Removing URLs
- Converting text to lowercase
- Removing unnecessary characters
- Removing stopwords
- Lemmatization
- Whitespace normalization

---

## 3. N-Gram Analysis

N-grams identify frequently occurring words and word combinations.

The system extracts:

- Unigrams
- Bigrams
- Trigrams

Examples include:

```text
waste money
waste time
highly recommend
great book
read book
```

N-gram analysis helps identify common expressions used by customers.

---

## 4. Keyword Extraction

Important terms and phrases are extracted using:

### TF-IDF

TF-IDF identifies terms that are important within the review collection.

### RAKE

RAKE extracts meaningful multi-word key phrases from individual reviews.

---

## 5. Named Entity Recognition

Named Entity Recognition identifies entities present in customer reviews.

Examples include:

- People
- Organizations
- Locations
- Dates
- Products
- Works of art

The project uses spaCy for NER.

---

## 6. Sentiment Analysis

Customer reviews are classified into:

- Positive
- Negative
- Neutral

The project uses VADER sentiment analysis along with product-review-specific NLP rules in the interactive dashboard.

The additional rules improve detection of expressions such as:

```text
very slow
poor quality
waste of money
does not work
not worth
do not buy
broken
defective
```

---

## 7. Aspect-Based Sentiment Analysis

Instead of analyzing only the overall sentiment of a review, the system identifies sentiment related to specific aspects.

Examples:

```text
Price
Quality
Battery
Design
Sound
Screen
Shipping
Performance
Product
Service
```

This allows the system to identify which specific aspects customers like or dislike.

---

## 8. Topic Extraction

Topic extraction identifies major themes discussed in the review collection.

Example themes include:

- Books
- Movies
- Products
- Music
- DVDs
- Games
- Product quality
- Purchasing experiences

---

## 9. Text Summarization

TextRank-based extractive summarization is used to generate concise summaries of longer reviews.

The system selects important sentences from the original review rather than generating completely new text.

---

## 10. Semantic Similarity

Reviews are converted into semantic embeddings using a pretrained Sentence Transformer.

The embeddings allow reviews to be compared according to their meaning.

This enables the system to identify reviews that express similar ideas even when different words are used.

---

## 11. Semantic Search

Semantic search allows users to search reviews using natural-language queries.

Example:

```text
customers complaining about poor product quality
```

Instead of searching only for the exact words, the system retrieves reviews with similar semantic meaning.

---

## 12. NLP Analytics & Visualization

The project generates visualizations including:

- Sentiment Distribution
- Topic Distribution
- Aspect Frequency
- Aspect Sentiment
- Review Length Distribution
- Top Words

These visualizations help users understand patterns across the review dataset.

---

# 🖥️ Interactive Dashboard

The final Streamlit dashboard provides:

1. Overview
2. Sentiment Analysis
3. Aspect-Based Sentiment
4. Topic Extraction
5. Semantic Search
6. Similar Reviews
7. Text Summarization
8. NLP Visualizations

---

# 📊 Dataset

## Amazon Review Polarity

The project uses the **Amazon Review Polarity dataset**.

A balanced working subset of 100,000 reviews is used.

| Category | Count |
|---|---:|
| Positive | 50,000 |
| Negative | 50,000 |
| Total | 100,000 |

---

# 📁 Project Structure

```text
nlp/
│
├── visuals/
│   ├── sentiment_distribution.png
│   ├── topic_distribution.png
│   ├── aspect_frequency.png
│   ├── aspect_sentiment.png
│   ├── review_length_distribution.png
│   └── top_words.png
│
├── amazon_review_aspects.csv
├── amazon_reviews_100k.csv
├── amazon_reviews_sentiment.csv
├── amazon_reviews_summarized.csv
├── amazon_reviews_topics.csv
├── aspect_sentiment_summary.csv
├── review_embeddings.csv
│
├── dashboard.py
├── run.py
├── pipeline.txt
└── README.md
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NLTK
- spaCy
- Scikit-learn
- VADER Sentiment
- RAKE
- Sentence Transformers
- Sumy / TextRank
- Matplotlib
- Seaborn
- WordCloud
- Streamlit

---

# 🧩 NLP Techniques

| Technique | Purpose |
|---|---|
| Text Preprocessing | Clean and normalize review text |
| N-Grams | Identify frequent word combinations |
| TF-IDF | Extract important terms |
| RAKE | Extract important phrases |
| NER | Identify named entities |
| Sentiment Analysis | Identify customer opinion |
| Aspect Sentiment | Analyze opinion for individual aspects |
| Topic Extraction | Discover major review themes |
| TextRank | Summarize reviews |
| Semantic Similarity | Find similar reviews |
| Semantic Search | Search reviews by meaning |
| Visualization | Present analytical insights |

---

# 🚀 Running the Project

Install the required libraries:

```powershell
pip install pandas nltk spacy scikit-learn matplotlib seaborn wordcloud rake-nltk vaderSentiment sentence-transformers sumy streamlit
```

Run the application:

```powershell
python run.py
```

The application starts the Streamlit dashboard and opens it in the browser.

Default address:

```text
http://localhost:8501
```

---

# 💡 Example Use Cases

### Customer Complaint Analysis

Identify reviews containing complaints about:

- Poor quality
- Slow performance
- Defective products
- High prices
- Bad service

### Product Aspect Analysis

Determine customer opinions about:

- Price
- Quality
- Battery
- Design
- Sound
- Performance
- Shipping

### Semantic Review Search

Search using natural-language questions or descriptions instead of exact keywords.

### Review Summarization

Convert lengthy customer reviews into concise summaries.

### Similar Review Discovery

Find reviews expressing similar experiences or opinions.

---

# ⚠️ Limitations

- VADER is lexicon-based and may struggle with highly contextual language.
- General-purpose NER models can sometimes incorrectly classify domain-specific product names.
- Semantic similarity depends on the pretrained embedding model.
- The project uses a 100,000-review working dataset rather than the complete original dataset.
- TextRank performs extractive rather than generative summarization.

---

# 🔮 Future Scope

Possible future improvements include:

- Product/category filtering
- Advanced aspect discovery
- Multilingual review analysis
- Interactive keyword exploration
- Review trend analysis
- Exportable analytical reports
- Cloud deployment of the dashboard

---

# 🏁 Conclusion

The Amazon Review NLP Intelligence System demonstrates how multiple Natural Language Processing techniques can be combined to transform unstructured customer reviews into useful insights.

The system goes beyond basic sentiment classification by incorporating keyword extraction, NER, aspect analysis, topic extraction, summarization, semantic similarity and semantic search.

The final interactive dashboard provides a single platform for exploring customer opinions, discovering review patterns and understanding customer feedback at scale.