"""Utility functions for NLP-Scraper project."""
import json
import ast
import logging
from typing import List, Any
import re
from urllib.parse import urlparse

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
    """Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text


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
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
