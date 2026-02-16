# NLP-Scraper Project

## Overview
This project builds an NLP-enriched News Intelligence platform that:
- Scrapes news articles from various sources
- Performs entity recognition (ORG)
- Classifies topics (Tech, Sports, Business, Entertainment, Politics)
- Analyzes sentiment (positive, negative, neutral)
- Detects environmental scandals related to companies
- Provides source analysis insights

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Download NLTK data: `python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"`

## Components
### Scraper (`scraper_news.py`)
- Collects at least 300 articles from news sources
- Stores unique ID, URL, date, headline, and body
- Saves data to CSV or SQL database

### NLP Engine (`nlp_enriched_news.py`)
- Performs entity detection (ORG) using SpaCy NER
- Topic classification with >95% accuracy using TF-IDF and Multinomial Naive Bayes
- Sentiment analysis using pre-trained VADER model from NLTK
- Scandal detection using sentence embeddings and cosine similarity
- Source analysis with visualizations
- Outputs enriched data to enhanced_news.csv

### Topic Classifier Training (`results/training_model.py`)
- Trains topic classification model with learning curves
- Achieves >95% accuracy on test data
- Saves trained model as topic_classifier.pkl

## Files
- `data/`: Storage for scraped articles
- `results/`: Output files, models, and visualizations
- `results/plots/`: Source analysis visualizations
- `requirements.txt`: Project dependencies
- `training_model.py`: Topic classification training code
- `learning_curves.png`: Model validation visualization
- `enhanced_news.csv`: Final enriched dataset

## Usage
1. Run scraper: `python scraper_news.py`
2. Process with NLP engine: `python nlp_enriched_news.py`

## Architecture

### Entity Detection
- Uses SpaCy's pre-trained NER model to detect ORG entities
- Processes both headline and body text
- Returns unique organizations mentioned in the article

### Topic Classification
- Uses TF-IDF vectorization with n-grams
- Implements Multinomial Naive Bayes classifier
- Achieves >95% accuracy on validation data
- Includes learning curves to verify model is not overfitted

### Sentiment Analysis
- Leverages pre-trained VADER sentiment analyzer from NLTK
- Returns compound sentiment score between -1 (most negative) and +1 (most positive)
- Categorizes sentiment as positive, negative, or neutral

### Scandal Detection
- Defines environmental disaster keywords (pollution, deforestation, etc.)
- Uses Sentence Transformers to compute embeddings
- Calculates cosine similarity between keyword embeddings and sentences containing entities
- Flags articles with highest similarity scores as potential scandals
- Returns distance metric and flag for top 10 articles

### Source Analysis (Optional)
- Generates daily insights:
  - Proportion of topics per day
  - Number of articles per day
  - Number of companies mentioned per day
  - Sentiment per day
- Generates company insights:
  - Companies mentioned the most
  - Sentiment per company
- Creates visualizations saved in results/plots/

## Output Format
The enhanced_news.csv contains the following columns:
- Unique_ID: UUID for each article
- URL: Original URL of the article
- Date: Publication date
- Headline: Article headline
- Body: Full article body
- Org: List of organizations mentioned
- Topics: Classified topic (Tech, Sports, Business, Entertainment, Politics)
- Sentiment: Compound sentiment score (-1 to +1)
- Scandal_distance: Distance metric for scandal detection
- Top_10: Boolean flag for top 10 scandal articles