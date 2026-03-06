# NLP-Scraper: News Intelligence Platform

## Overview

This project builds an **NLP-enriched News Intelligence platform** that helps analysts detect relevant information from news articles. The platform connects to news data sources, performs entity detection, topic classification, sentiment analysis, and scandal detection.

### Key Features

- **Web Scraping**: Collects 300+ articles from major news RSS feeds
- **Entity Detection**: Identifies organizations (ORG) using SpaCy NER
- **Topic Classification**: Categorizes articles into Tech, Sports, Business, Entertainment, or Politics (>95% accuracy)
- **Sentiment Analysis**: Analyzes sentiment using pre-trained VADER model (NLTK)
- **Scandal Detection**: Detects potential environmental scandals using sentence embeddings
- **Source Analysis**: Generates visualizations about news sources and trends

---

## Project Structure

```
NLP-Scraper/
├── data/
│   └── scraped_articles.csv      # Raw scraped articles
├── results/
│   ├── training_model.py         # Topic classifier training script
│   ├── topic_classifier.pkl      # Trained model (generated)
│   ├── learning_curves.png       # Model validation plot (generated)
│   ├── confusion_matrix.png      # Performance breakdown (generated)
│   ├── enhanced_news.csv         # Enriched output (generated)
│   └── plots/                    # Source analysis visualizations
│       ├── topics_per_day.png
│       ├── articles_per_day.png
│       ├── companies_mentioned_per_day.png
│       ├── sentiment_per_day.png
│       ├── top_companies.png
│       └── sentiment_per_company.png
├── scraper_news.py               # News scraper module
├── nlp_enriched_news.py          # NLP processing engine
├── utils.py                      # Text preprocessing utilities
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLP Models and Data

```bash
# Download SpaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt')"
```

### 3. Verify Installation

```bash
python -c "import spacy, nltk, sklearn, sentence_transformers; print('All dependencies installed successfully!')"
```

---

## Usage

### Step 1: Scrape News Articles

```bash
python scraper_news.py
```

**Output:**
```
1. scraping https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml
        requesting ...
        parsing ...
        saved in data/scraped_articles.csv
        Collected 1/300

2. scraping https://feeds.bbci.co.uk/news/rss.xml
        requesting ...
        parsing ...
        ...

Successfully scraped and saved 300 articles to data/scraped_articles.csv
```

**Stored Data:**
- `unique_id`: UUID for each article
- `url`: Article URL
- `date`: Publication date
- `headline`: Article headline
- `body`: Full article body

### Step 2: Train Topic Classifier (Optional)

```bash
python results/training_model.py
```

**Output:**
- `results/topic_classifier.pkl` - Trained model
- `results/learning_curves.png` - Learning curves visualization
- `results/confusion_matrix.png` - Performance breakdown

### Step 3: Run NLP Engine

```bash
python nlp_enriched_news.py
```

**Output:**
```
Enriching https://www.nytimes.com/2024/...:

Cleaning document ...

---------- Detect entities ----------
Detected 5 companies which are Apple, Google, Microsoft...

---------- Topic detection ----------
Text preprocessing ...
The topic of the article is: Technology

---------- Sentiment analysis ----------
Text preprocessing ...
The article 'Apple announces new iPhone...' has a positive sentiment (score: 0.723)

---------- Scandal detection ----------
Computing embeddings and distance ...

Successfully processed 300 articles
Enhanced news data saved to results/enhanced_news.csv
```

---

## Output Format

The `enhanced_news.csv` contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `Unique_ID` | UUID | Unique identifier for each article |
| `URL` | str | Original article URL |
| `Date` | date | Publication/scraping date |
| `Headline` | str | Article headline |
| `Body` | str | Full article body text |
| `Org` | list[str] | Detected organizations (companies) |
| `Topics` | str | Classified topic (Tech/Sports/Business/Entertainment/Politics) |
| `Sentiment` | float | Compound sentiment score (-1.0 to +1.0) |
| `Scandal_distance` | float | Distance metric for scandal detection |
| `Top_10` | bool | Flag for top 10 potential scandal articles |

---

## NLP Pipeline Details

### 1. Entity Detection (ORG)

**Method:** SpaCy Named Entity Recognition

- Uses pre-trained `en_core_web_sm` model
- Detects organization entities (companies, institutions)
- Processes both headline and body text
- Returns unique organizations per article

**Learning Objective:** Named Entity Recognition with NLTK and SpaCy

### 2. Topic Classification

**Method:** TF-IDF + Multinomial Naive Bayes

- **Vectorization:** TF-IDF with unigrams and bigrams (max 10,000 features)
- **Classifier:** Multinomial Naive Bayes (alpha=0.5)
- **Target Accuracy:** >95% on test data
- **Topics:** Technology, Sports, Business, Entertainment, Politics

**Learning Curves:** The `learning_curves.png` plot shows:
- Training accuracy vs. validation accuracy
- Confirms model is not overfitting
- Validates generalization capability

### 3. Sentiment Analysis

**Method:** Pre-trained VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Why Pre-trained:**
1. Industry best practice - leverage existing high-quality models
2. Labeled news sentiment data is expensive
3. VADER is optimized for social media and news text

**Output:**
- Compound score: -1.0 (most negative) to +1.0 (most positive)
- Classification:
  - Positive: compound ≥ 0.05
  - Negative: compound ≤ -0.05
  - Neutral: -0.05 < compound < 0.05

### 4. Scandal Detection

**Goal:** Detect potential environmental disasters related to detected companies.

#### Methodology

**Step 1: Define Environmental Keywords**

Carefully selected keywords specific to environmental disasters:
- `oil spill`, `toxic waste`, `chemical leak`
- `water contamination`, `soil contamination`
- `hazardous waste`, `nuclear leak`
- `industrial pollution`, `waste dumping`
- `ecological disaster`, etc.

**Keyword Selection Criteria:**
- Specific to environmental issues
- Minimize false positives (avoid ambiguous words)
- Cover various types of environmental disasters

**Step 2: Compute Embeddings**

**Embedding Model:** `all-MiniLM-L6-v2` (Sentence Transformers)

**Why this model:**
- **Speed:** Fast inference (important for live processing)
- **Size:** Lightweight (~80MB)
- **Quality:** Good balance of accuracy and efficiency
- **Semantic Understanding:** Captures meaning beyond keyword matching
- **Pre-trained:** No fine-tuning required

**Step 3: Calculate Distance/Similarity**

**Distance Metric:** Cosine Distance = `1 - cosine_similarity`

**Why Cosine Similarity:**
- **Standard for NLP:** Industry standard for text similarity
- **Magnitude Invariant:** Focuses on angle, not vector magnitude
- **Normalized:** Works well with normalized embeddings
- **Interpretable:** 1 = identical, 0 = orthogonal, -1 = opposite

**Calculation:**
```python
# For each sentence containing an organization:
similarity = cos(sentence_embedding, keyword_embedding)
distance = 1 - similarity

# Average distance across all relevant sentences
avg_distance = mean(all_distances)

# Flag as scandal if distance < threshold
is_scandal = avg_distance < (1 - SCANDAL_SIMILARITY_THRESHOLD)
```

**Threshold:** 0.65 cosine similarity (distance < 0.35)

**Step 4: Flag Top Articles**

- Rank articles by scandal distance (lowest = most similar)
- Flag top 10 articles as potential scandals
- Store `Scandal_distance` metric for each article

---

## Source Analysis (Optional)

Generates insights from scraped data (requires 5+ days of data):

### Daily Metrics
- **Topics per Day:** Stacked bar chart of topic distribution
- **Articles per Day:** Line chart of article volume
- **Companies per Day:** Line chart of organization mentions
- **Sentiment per Day:** Line chart of average sentiment

### Company Metrics
- **Top 10 Companies:** Bar chart of most mentioned organizations
- **Sentiment per Company:** Bar chart of average sentiment by company

All plots saved to `results/plots/`

---

## Learning Objectives Covered

### NLP Preprocessing
- ✅ Text tokenization (sentence and word level)
- ✅ Stopword removal using NLTK
- ✅ Stemming using Porter Stemmer
- ✅ Complete preprocessing pipeline

### Machine Learning
- ✅ Bag-of-words representation (TF-IDF)
- ✅ Model training and evaluation
- ✅ Learning curves for overfitting detection
- ✅ Cross-validation

### NLP Techniques
- ✅ Named Entity Recognition (SpaCy)
- ✅ Sentiment analysis (pre-trained model)
- ✅ Text embeddings (Sentence Transformers)
- ✅ Semantic similarity (Cosine similarity)

---

## Model Performance

### Topic Classifier
- **Accuracy:** >95% on test data
- **Classes:** 5 (Technology, Sports, Business, Entertainment, Politics)
- **Validation:** Learning curves confirm no overfitting
- **Reproducibility:** Fixed random state (42)

### Sentiment Analyzer
- **Model:** VADER (pre-trained)
- **Range:** -1.0 to +1.0 (compound score)
- **Optimized for:** Social media and news text

### Scandal Detection
- **Embeddings:** all-MiniLM-L6-v2
- **Similarity Metric:** Cosine similarity
- **Threshold:** 0.65 (configurable)
- **Output:** Distance metric + Top 10 flag

---

## Configuration

Edit `config.py` to customize:

```python
# Number of articles to scrape
MIN_ARTICLES = 300

# Topic classification accuracy target
MODEL_ACCURACY_THRESHOLD = 0.95

# Scandal detection sensitivity
SCANDAL_SIMILARITY_THRESHOLD = 0.65

# Sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05
```

---

## Troubleshooting

### SpaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Not Found
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
```

### Insufficient Articles Scraped
- Check internet connection
- Verify RSS feed URLs are accessible
- Increase `REQUEST_TIMEOUT` in config.py
- Some feeds may be temporarily unavailable

### Model Accuracy Below 95%
- Run `training_model.py` to retrain with fresh data
- Increase `TRAINING_DATA_MULTIPLIER` in config.py
- Check `learning_curves.png` for overfitting signs

---

## License

This project is for educational purposes as part of the AI branch curriculum.

---

## Author

Natural Language Processing (NLP) Specialist
Tech Startup - Sentiment Analysis Tool Development
