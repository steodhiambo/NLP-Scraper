import pandas as pd
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
import os
from collections import Counter

from config import (
    SPACY_MODEL, SENTENCE_TRANSFORMER_MODEL, SCANDAL_KEYWORDS,
    SCANDAL_SIMILARITY_THRESHOLD, TOP_SCANDAL_COUNT, TOPICS,
    TFIDF_MAX_FEATURES, MODEL_ACCURACY_THRESHOLD, TOPIC_CLASSIFIER_MODEL,
    SCRAPED_ARTICLES_CSV, ENHANCED_NEWS_CSV, PLOTS_DIR,
    SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD
)
from utils import setup_logging, safe_parse_list, extract_domain, clean_text

setup_logging()
logger = logging.getLogger(__name__)

class NLPEngine:
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.sia = self._load_sentiment_analyzer()
        self.sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        self.scandal_keywords = SCANDAL_KEYWORDS
        self.keyword_embeddings = self.sentence_model.encode(self.scandal_keywords)
        self.topic_classifier = None
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling."""
        try:
            return spacy.load(SPACY_MODEL)
        except OSError:
            logger.error(f"SpaCy model '{SPACY_MODEL}' not found. Install: python -m spacy download {SPACY_MODEL}")
            raise
    
    def _load_sentiment_analyzer(self):
        """Load VADER sentiment analyzer with error handling."""
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            logger.info("Downloading VADER lexicon...")
            nltk.download('vader_lexicon')
            return SentimentIntensityAnalyzer()
    
    def detect_entities(self, text):
        """Detect ORG entities in text using SpaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of unique organization names
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        return list(dict.fromkeys(org_entities))
    
    def preprocess_text(self, text):
        """Basic text preprocessing.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        return clean_text(text)
    
    def load_topic_classifier(self, model_path=None):
        """Load pre-trained topic classifier.
        
        Args:
            model_path: Path to saved model
        """
        if model_path is None:
            model_path = TOPIC_CLASSIFIER_MODEL
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.topic_classifier = pickle.load(f)
            logger.info(f"Loaded topic classifier from {model_path}")
        else:
            logger.warning(f"Topic classifier not found at {model_path}. Training new model...")
            self.train_topic_classifier()
    
    def train_topic_classifier(self, data_path='data/topic_training_data.csv'):
        """Train topic classification model"""
        logger.info("Training topic classification model...")
        
        # For demonstration purposes, we'll create sample training data
        # In a real scenario, you would load this from a file
        sample_topics = [
            ("Apple announces new iPhone with revolutionary features", "Technology"),
            ("Google's new AI breakthrough changes everything", "Technology"),
            ("Microsoft releases Windows 11 update", "Technology"),
            ("Tesla stock price reaches new high", "Business"),
            ("Amazon reports record profits", "Business"),
            ("New banking regulations announced", "Business"),
            ("Champions League final scheduled for May", "Sports"),
            ("Olympics postponed due to pandemic", "Sports"),
            ("World Cup preparations underway", "Sports"),
            ("Hollywood movie wins multiple awards", "Entertainment"),
            ("New streaming service launches", "Entertainment"),
            ("Celebrity couple announces divorce", "Entertainment"),
            ("Government passes new legislation", "Politics"),
            ("Election results announced", "Politics"),
            ("International summit addresses climate change", "Politics")
        ] * 20  # Multiply to have more samples
        
        # Create training data
        texts = [item[0] for item in sample_topics]
        labels = [item[1] for item in sample_topics]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        self.topic_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        self.topic_classifier.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.topic_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Topic classifier accuracy: {accuracy:.4f}")
        
        if accuracy < MODEL_ACCURACY_THRESHOLD:
            logger.warning(f"Model accuracy ({accuracy:.4f}) is below {MODEL_ACCURACY_THRESHOLD}. Consider improving.")
        else:
            logger.info(f"Model meets accuracy requirement (>{MODEL_ACCURACY_THRESHOLD})")
        
        with open(TOPIC_CLASSIFIER_MODEL, 'wb') as f:
            pickle.dump(self.topic_classifier, f)
        
        # Plot learning curves
        self.plot_learning_curves(X_train, X_test, y_train, y_test)
        
        return self.topic_classifier
    
    def plot_learning_curves(self, X_train, X_test, y_train, y_test):
        """Plot learning curves to validate model training"""
        from sklearn.model_selection import learning_curve
        
        estimator = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.title('Learning Curves for Topic Classification Model')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the plot
        plt.savefig('results/learning_curves.png')
        plt.close()
        
        logger.info("Learning curves saved to results/learning_curves.png")
    
    def classify_topic(self, text):
        """Classify the topic of text.
        
        Args:
            text: Input text
            
        Returns:
            Predicted topic
        """
        if self.topic_classifier is None:
            self.load_topic_classifier()
        
        return self.topic_classifier.predict([text])[0]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, scores_dict)
        """
        scores = self.sia.polarity_scores(text)
        
        if scores['compound'] >= SENTIMENT_POSITIVE_THRESHOLD:
            sentiment = 'positive'
        elif scores['compound'] <= SENTIMENT_NEGATIVE_THRESHOLD:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return sentiment, scores
    
    def detect_scandal(self, text, org_entities):
        """Detect potential scandals related to organizations.
        
        Args:
            text: Article text
            org_entities: List of organization names
            
        Returns:
            Tuple of (distance, is_scandal)
        """
        if not org_entities or not text:
            return 0.0, False
        
        # Find sentences that contain organization names
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        relevant_sentences = []
        
        for sentence in sentences:
            for org in org_entities:
                if org.lower() in sentence.lower():
                    relevant_sentences.append(sentence)
                    break  # Found an org in this sentence, move to next sentence
        
        if not relevant_sentences:
            return 0.0, False
        
        # Compute embeddings for relevant sentences
        sentence_embeddings = self.sentence_model.encode(relevant_sentences)
        
        # Calculate distances between sentence embeddings and keyword embeddings
        min_distances = []
        
        for sent_emb in sentence_embeddings:
            # Calculate cosine similarity between sentence and each keyword
            similarities = np.dot(self.keyword_embeddings, sent_emb) / (
                np.linalg.norm(self.keyword_embeddings, axis=1) * np.linalg.norm(sent_emb)
            )
            
            # Take the maximum similarity (minimum distance) for this sentence
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
            min_distances.append(1 - max_similarity)  # Convert similarity to distance
        
        avg_distance = np.mean(min_distances) if min_distances else 1.0
        threshold = SCANDAL_SIMILARITY_THRESHOLD
        is_scandal = avg_distance < (1 - threshold)
        
        return avg_distance, is_scandal
    
    def extract_domain_from_url(self, url):
        """Extract domain from URL.
        
        Args:
            url: Full URL
            
        Returns:
            Domain name
        """
        return extract_domain(url)
    
    def perform_source_analysis(self, df):
        """Perform source analysis to generate insights"""
        logger.info("Performing source analysis...")
        
        # Convert date column to datetime if it's not already
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        
        df['Domain'] = df['URL'].apply(self.extract_domain_from_url)
        
        PLOTS_DIR.mkdir(exist_ok=True)
        
        # 1. Proportion of topics per day
        daily_topic_counts = df.groupby([df['Date_parsed'].dt.date, 'Topics']).size().unstack(fill_value=0)
        plt.figure(figsize=(12, 8))
        daily_topic_counts.plot(kind='bar', stacked=True)
        plt.title('Proportion of Topics per Day')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'topics_per_day.png')
        plt.close()
        
        # 2. Number of articles per day
        daily_article_count = df.groupby(df['Date_parsed'].dt.date).size()
        plt.figure(figsize=(12, 6))
        plt.plot(daily_article_count.index, daily_article_count.values, marker='o')
        plt.title('Number of Articles per Day')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'articles_per_day.png')
        plt.close()
        
        # 3. Number of companies mentioned per day
        df['Num_Orgs'] = df['Org'].apply(lambda x: len(safe_parse_list(x)))
        daily_org_count = df.groupby(df['Date_parsed'].dt.date)['Num_Orgs'].sum()
        plt.figure(figsize=(12, 6))
        plt.plot(daily_org_count.index, daily_org_count.values, marker='o', color='orange')
        plt.title('Number of Companies Mentioned per Day')
        plt.xlabel('Date')
        plt.ylabel('Number of Companies')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'companies_mentioned_per_day.png')
        plt.close()
        
        # 4. Sentiment per day
        daily_sentiment = df.groupby(df['Date_parsed'].dt.date)['Sentiment'].mean()
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o', color='green')
        plt.title('Average Sentiment per Day')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'sentiment_per_day.png')
        plt.close()
        
        all_orgs = []
        for org_list_str in df['Org']:
            org_list = safe_parse_list(org_list_str)
            if org_list:
                all_orgs.extend(org_list)
        
        org_counts = Counter(all_orgs)
        top_companies = dict(org_counts.most_common(10))
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_companies.keys(), top_companies.values())
        plt.title('Top 10 Companies Mentioned')
        plt.xlabel('Companies')
        plt.ylabel('Number of Mentions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'top_companies.png')
        plt.close()
        
        company_sentiments = {}
        for company in top_companies.keys():
            company_rows = df[df['Org'].apply(lambda x: company in safe_parse_list(x))]
            if not company_rows.empty:
                company_sentiments[company] = company_rows['Sentiment'].mean()
        
        if company_sentiments:
            plt.figure(figsize=(12, 6))
            plt.bar(company_sentiments.keys(), company_sentiments.values(), color='purple')
            plt.title('Average Sentiment per Company (Top 10)')
            plt.xlabel('Companies')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'sentiment_per_company.png')
            plt.close()
        
        logger.info(f"Source analysis completed. Plots saved to {PLOTS_DIR}")
    
    def process_articles(self, input_file=None, output_file=None):
        """Process all articles and enrich with NLP insights.
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
            
        Returns:
            Enriched DataFrame
        """
        if input_file is None:
            input_file = SCRAPED_ARTICLES_CSV
        if output_file is None:
            output_file = ENHANCED_NEWS_CSV
        
        logger.info(f"Loading articles from {input_file}")
        
        # Load articles
        df = pd.read_csv(input_file)
        
        # Prepare columns for enriched data
        enriched_data = {
            'Unique_ID': [],
            'URL': [],
            'Date': [],
            'Headline': [],
            'Body': [],
            'Org': [],
            'Topics': [],
            'Sentiment': [],
            'Scandal_distance': [],
            'Top_10': []
        }
        
        total_articles = len(df)
        
        for idx, row in df.iterrows():
            logger.info(f"Processing article {idx+1}/{total_articles}: {row['headline'][:50]}...")
            
            # Combine headline and body for comprehensive analysis
            full_text = f"{row['headline']} {row['body']}"
            
            print(f"\nEnriching {row['url']}:")
            
            # Clean document (optional)
            print("Cleaning document ...")
            cleaned_text = self.preprocess_text(full_text)
            
            # Detect entities
            print("---------- Detect entities ----------")
            org_entities = self.detect_entities(full_text)
            print(f"Detected {len(org_entities)} companies which are {', '.join(org_entities[:3])}{'...' if len(org_entities) > 3 else ''}")
            
            # Topic detection
            print("---------- Topic detection ----------")
            print("Text preprocessing ...")
            topic = self.classify_topic(full_text)
            print(f"The topic of the article is: {topic}")
            
            # Sentiment analysis
            print("---------- Sentiment analysis ----------")
            print("Text preprocessing ...")
            sentiment, sentiment_scores = self.analyze_sentiment(full_text)
            print(f"The article '{row['headline'][:30]}{'...' if len(row['headline']) > 30 else ''}' has a {sentiment} sentiment")
            
            # Scandal detection
            print("---------- Scandal detection ----------")
            print("Computing embeddings and distance ...")
            scandal_distance, is_scandal = self.detect_scandal(full_text, org_entities)
            if is_scandal:
                print(f"Environmental scandal detected for {', '.join(org_entities[:2])}{'...' if len(org_entities) > 2 else ''}")
            
            # Store enriched data
            enriched_data['Unique_ID'].append(row['id'])
            enriched_data['URL'].append(row['url'])
            enriched_data['Date'].append(row['date'])
            enriched_data['Headline'].append(row['headline'])
            enriched_data['Body'].append(row['body'])
            enriched_data['Org'].append(str(org_entities))  # Convert to string for CSV
            enriched_data['Topics'].append(topic)
            enriched_data['Sentiment'].append(sentiment_scores['compound'])
            enriched_data['Scandal_distance'].append(scandal_distance)
            enriched_data['Top_10'].append(False)  # Will be set later for top 10 scandal articles
        
        enriched_df = pd.DataFrame(enriched_data)
        
        top_10_indices = enriched_df.nsmallest(TOP_SCANDAL_COUNT, 'Scandal_distance').index
        enriched_df.loc[top_10_indices, 'Top_10'] = True
        
        enriched_df.to_csv(output_file, index=False)
        logger.info(f"Enhanced news data saved to {output_file}")
        
        # Perform source analysis (optional)
        self.perform_source_analysis(enriched_df)
        
        return enriched_df


def main():
    """Main entry point for NLP engine."""
    try:
        nlp_engine = NLPEngine()
        enriched_df = nlp_engine.process_articles()
        print(f"\nSuccessfully processed and saved enriched news data to {ENHANCED_NEWS_CSV}")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()