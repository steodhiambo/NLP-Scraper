"""
Utility functions for NLP-Scraper project.

Provides text preprocessing, parsing, and helper functions
for the NLP pipeline.
"""

import json
import ast
import logging
from typing import List, Any
import re
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


logger = logging.getLogger(__name__)


def safe_parse_list(value: Any) -> List:
    """Safely parse a string representation of a list.

    Args:
        value: String or list to parse

    Returns:
        Parsed list or empty list if parsing fails
    """
    if isinstance(value, list):
        return value
    if not value or value == '[]':
        return []

    try:
        # Try JSON first (safer)
        return json.loads(value.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError):
        try:
            # Fallback to ast.literal_eval (safer than eval)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse list: {value}")
            return []


def extract_domain(url: str) -> str:
    """Extract domain from URL.

    Args:
        url: Full URL string

    Returns:
        Domain name or 'unknown'
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.replace('www.', '')
    except Exception as e:
        logger.warning(f"Failed to extract domain from {url}: {e}")
        return 'unknown'


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters and punctuation
    3. Remove extra whitespace
    
    This is the basic cleaning function for preprocessing.

    Args:
        text: Raw text string

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_text_full(text: str, remove_stopwords: bool = True, 
                         apply_stemming: bool = True) -> str:
    """
    Complete text preprocessing pipeline for NLP tasks.
    
    This function implements the full preprocessing pipeline:
    1. Lowercase conversion
    2. Punctuation and special character removal
    3. Tokenization (word level)
    4. Stopword removal (optional)
    5. Stemming (optional)
    
    Learning Objectives covered:
    - Text tokenization at word level
    - Stopword removal using NLTK
    - Stemming using Porter Stemmer
    
    Args:
        text: Raw input text
        remove_stopwords: Whether to remove English stopwords (default: True)
        apply_stemming: Whether to apply Porter stemming (default: True)
        
    Returns:
        Preprocessed text string
    """
    if not text:
        return ""
    
    # Step 1: Basic cleaning (lowercase, remove punctuation)
    text = clean_text(text)
    
    # Step 2: Tokenization at word level
    words = text.split()
    
    # Step 3: Remove stopwords if enabled
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        except LookupError:
            logger.info("Downloading stopwords corpus...")
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
    
    # Step 4: Apply stemming if enabled
    if apply_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]
    
    # Join tokens back to string
    return ' '.join(words)


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences.
    
    Learning Objective: Text tokenization at sentence level
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        logger.info("Downloading punkt tokenizer...")
        nltk.download('punkt')
        sentences = sent_tokenize(text)
    
    return sentences


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words using NLTK.
    
    Learning Objective: Text tokenization at word level
    
    Args:
        text: Input text
        
    Returns:
        List of words
    """
    if not text:
        return []
    
    try:
        words = word_tokenize(text)
    except LookupError:
        logger.info("Downloading punkt tokenizer...")
        nltk.download('punkt')
        words = word_tokenize(text)
    
    return words


def remove_stopwords_from_list(words: List[str]) -> List[str]:
    """
    Remove English stopwords from a list of words.
    
    Learning Objective: Stopword removal
    
    Args:
        words: List of words
        
    Returns:
        List of words with stopwords removed
    """
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        logger.info("Downloading stopwords corpus...")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    return [w for w in words if w not in stop_words]


def stem_words(words: List[str]) -> List[str]:
    """
    Apply Porter stemming to a list of words.
    
    Learning Objective: Stemming
    
    Args:
        words: List of words
        
    Returns:
        List of stemmed words
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in words]


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if not text:
        return ""
    return text[:max_length] if len(text) > max_length else text


def setup_logging(log_file: str = None, level: str = 'INFO'):
    """Setup logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
