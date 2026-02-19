"""Configuration settings for NLP-Scraper project."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Scraper settings
RSS_FEEDS = [
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://rss.cnn.com/rss/edition.rss'
]

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
REQUEST_TIMEOUT = 30
MIN_ARTICLES = int(os.getenv('MIN_ARTICLES', 300))
SCRAPE_DELAY_MIN = 0.5
SCRAPE_DELAY_MAX = 1.5
MAX_RETRIES = 3
MAX_CONTENT_LENGTH = 5000

# File paths
SCRAPED_ARTICLES_CSV = DATA_DIR / "scraped_articles.csv"
SCRAPED_ARTICLES_DB = DATA_DIR / "scraped_articles.db"
ENHANCED_NEWS_CSV = RESULTS_DIR / "enhanced_news.csv"
TOPIC_CLASSIFIER_MODEL = RESULTS_DIR / "topic_classifier.pkl"
LEARNING_CURVES_PLOT = RESULTS_DIR / "learning_curves.png"

# NLP settings
SPACY_MODEL = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Topic classification
TOPICS = ['Technology', 'Sports', 'Business', 'Entertainment', 'Politics']
TFIDF_MAX_FEATURES = 10000
MODEL_ACCURACY_THRESHOLD = 0.95

# Sentiment analysis
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05

# Scandal detection
SCANDAL_KEYWORDS = [
    "pollution", "deforestation", "oil spill", "toxic waste", "chemical leak",
    "environmental disaster", "water contamination", "air pollution", 
    "soil contamination", "ecological damage", "habitat destruction", 
    "species extinction", "climate change", "carbon emissions", 
    "greenhouse gas", "environmental violation", "illegal dumping",
    "environmental crime", "environmental catastrophe", "environmental emergency"
]
SCANDAL_SIMILARITY_THRESHOLD = 0.7
TOP_SCANDAL_COUNT = 10

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / "nlp_scraper.log"
