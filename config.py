"""
Configuration settings for NLP-Scraper project.

Centralized configuration for scraper, NLP engine, and model training.
"""

import os
from pathlib import Path

# =============================================================================
# Base directories
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Scraper settings
# =============================================================================
# RSS feeds from major news sources (easy to scrape, frequently updated)
RSS_FEEDS = [
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://rss.cnn.com/rss/edition.rss',
    'https://www.aljazeera.com/xml/rss/all.xml',
    'https://feeds.reuters.com/reuters/topNews'
]

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
REQUEST_TIMEOUT = 30  # seconds
MIN_ARTICLES = int(os.getenv('MIN_ARTICLES', 300))  # Target: at least 300 articles
SCRAPE_DELAY_MIN = 0.5  # seconds between requests
SCRAPE_DELAY_MAX = 1.5  # seconds between requests
MAX_RETRIES = 3  # for failed requests
MAX_CONTENT_LENGTH = 10000  # maximum characters per article body

# File paths for scraper output
SCRAPED_ARTICLES_CSV = DATA_DIR / "scraped_articles.csv"
SCRAPED_ARTICLES_DB = DATA_DIR / "scraped_articles.db"

# =============================================================================
# NLP Engine settings
# =============================================================================
# SpaCy model for Named Entity Recognition
SPACY_MODEL = "en_core_web_sm"

# Sentence Transformer model for embeddings
# all-MiniLM-L6-v2: Fast, lightweight, good for semantic similarity
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# File paths for NLP output
ENHANCED_NEWS_CSV = RESULTS_DIR / "enhanced_news.csv"
TOPIC_CLASSIFIER_MODEL = RESULTS_DIR / "topic_classifier.pkl"
LEARNING_CURVES_PLOT = RESULTS_DIR / "learning_curves.png"

# =============================================================================
# Topic Classification settings
# =============================================================================
# Target topics for classification
TOPICS = ['Technology', 'Sports', 'Business', 'Entertainment', 'Politics']

# TF-IDF vectorizer settings
TFIDF_MAX_FEATURES = 10000

# Model performance threshold
MODEL_ACCURACY_THRESHOLD = 0.95  # Target: >95% accuracy on test data

# =============================================================================
# Sentiment Analysis settings
# =============================================================================
# VADER sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.05  # compound >= 0.05 -> positive
SENTIMENT_NEGATIVE_THRESHOLD = -0.05  # compound <= -0.05 -> negative
# Between -0.05 and 0.05 -> neutral

# =============================================================================
# Scandal Detection settings
# =============================================================================
# Environmental disaster keywords
# Carefully selected to avoid false positives:
# - Specific to environmental issues
# - Less likely to appear in non-environmental contexts
SCANDAL_KEYWORDS = [
    "oil spill",
    "toxic waste",
    "chemical leak",
    "water contamination",
    "soil contamination", 
    "hazardous waste",
    "nuclear leak",
    "radiation leak",
    "industrial pollution",
    "waste dumping",
    "illegal dumping",
    "environmental contamination",
    "toxic spill",
    "petroleum spill",
    "gas leak",
    "pipeline leak",
    "chemical spill",
    "sewage contamination",
    "groundwater contamination",
    "ecological disaster"
]

# Cosine similarity threshold for scandal detection
# Higher threshold = more strict (fewer false positives)
SCANDAL_SIMILARITY_THRESHOLD = 0.65

# Number of top scandal articles to flag
TOP_SCANDAL_COUNT = 10

# =============================================================================
# Logging settings
# =============================================================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / "nlp_scraper.log"

# =============================================================================
# Model Training settings
# =============================================================================
# Random state for reproducibility
RANDOM_STATE = 42

# Cross-validation folds for learning curves
CV_FOLDS = 5

# Training data multiplier (for synthetic data generation)
TRAINING_DATA_MULTIPLIER = 30
