import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
from datetime import datetime
import random
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    RSS_FEEDS, USER_AGENT, REQUEST_TIMEOUT, MIN_ARTICLES,
    SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX, MAX_RETRIES,
    MAX_CONTENT_LENGTH, SCRAPED_ARTICLES_CSV, SCRAPED_ARTICLES_DB
)
from utils import setup_logging, truncate_text

setup_logging()
logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self):
        self.sources = RSS_FEEDS
        self.headers = {'User-Agent': USER_AGENT}
        self.articles = []
        self.session = self._create_session()
    
    def _create_session(self):
        """Create requests session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def fetch_rss_feed(self, url):
        """Fetch articles from RSS feed.
        
        Args:
            url: RSS feed URL
            
        Returns:
            List of article dictionaries
        """
        try:
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml-xml')
            items = soup.find_all('item')
            
            articles = []
            for item in items:
                title = item.find('title').text if item.find('title') else ''
                link = item.find('link').text if item.find('link') else ''
                pub_date = item.find('pubdate').text if item.find('pubdate') else ''
                
                # Skip if no link or title
                if not link or not title:
                    continue
                
                articles.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date
                })
            
            logger.info(f"Fetched {len(articles)} articles from {url}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {str(e)}")
            return []
    
    def scrape_article_content(self, url):
        """Scrape the full content of an article.
        
        Args:
            url: Article URL
            
        Returns:
            Article content text
        """
        try:
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Try to find article content - this varies by site
            # Common selectors for article content
            selectors = [
                'article', '.article-body', '.story-body', '.post-content',
                '.entry-content', 'main', '.main-content', '.content'
            ]
            
            content = ""
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(strip=True, separator=' ')
                    break
            
            # If no specific content found, get all text
            if not content:
                content = soup.get_text(strip=True, separator=' ')
            
            content = truncate_text(content, MAX_CONTENT_LENGTH)
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {str(e)}")
            return ""
    
    def scrape_news(self, min_articles=None):
        """Main scraping function to collect articles.
        
        Args:
            min_articles: Minimum number of articles to collect
            
        Returns:
            List of scraped articles
        """
        if min_articles is None:
            min_articles = MIN_ARTICLES
        
        logger.info(f"Starting to scrape at least {min_articles} articles...")
        
        articles_collected = 0
        
        while articles_collected < min_articles:
            for source in self.sources:
                logger.info(f"Scraping from source: {source}")
                
                rss_articles = self.fetch_rss_feed(source)
                
                for rss_article in rss_articles:
                    if articles_collected >= min_articles:
                        break
                    
                    logger.info(f"Scraping article: {rss_article['title'][:50]}...")
                    
                    # Generate unique ID
                    article_id = str(uuid.uuid4())
                    
                    # Scrape full content
                    body = self.scrape_article_content(rss_article['link'])
                    
                    if body:  # Only store if we got content
                        article_data = {
                            'id': article_id,
                            'url': rss_article['link'],
                            'date': rss_article.get('pub_date', datetime.now().isoformat()),
                            'headline': rss_article['title'],
                            'body': body
                        }
                        
                        self.articles.append(article_data)
                        articles_collected += 1
                        
                        logger.info(f"Collected article {articles_collected}/{min_articles}: {rss_article['title'][:30]}...")
                    
                    time.sleep(random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX))
                
                if articles_collected >= min_articles:
                    break
            
            # If we haven't collected enough articles, try again
            if articles_collected < min_articles:
                logger.info(f"Still need {min_articles - articles_collected} more articles. Waiting before next round...")
                time.sleep(10)  # Wait before trying again
        
        logger.info(f"Completed scraping {len(self.articles)} articles")
        return self.articles
    
    def save_to_csv(self, filename=None):
        """Save articles to CSV file.
        
        Args:
            filename: Output CSV file path
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = SCRAPED_ARTICLES_CSV
        
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(self.articles)} articles to {filename}")
        return filename
    
    def save_to_sql(self, db_filename=None):
        """Save articles to SQLite database.
        
        Args:
            db_filename: Output database file path
            
        Returns:
            Path to saved database
        """
        import sqlite3
        
        if db_filename is None:
            db_filename = SCRAPED_ARTICLES_DB
        
        conn = sqlite3.connect(db_filename)
        
        # Create table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                url TEXT,
                date TEXT,
                headline TEXT,
                body TEXT
            )
        ''')
        
        # Insert articles
        for article in self.articles:
            conn.execute(
                'INSERT OR REPLACE INTO articles (id, url, date, headline, body) VALUES (?, ?, ?, ?, ?)',
                (article['id'], article['url'], article['date'], article['headline'], article['body'])
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(self.articles)} articles to {db_filename}")
        return db_filename


def main():
    """Main entry point for scraper."""
    try:
        scraper = NewsScraper()
        articles = scraper.scrape_news()
        filename = scraper.save_to_csv()
        print(f"Successfully scraped and saved {len(articles)} articles to {filename}")
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()