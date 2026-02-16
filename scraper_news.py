import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
from datetime import datetime, timedelta
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self):
        # Example news sources - in practice, you'd want to use RSS feeds or APIs
        self.sources = [
            'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://rss.cnn.com/rss/edition.rss'
        ]
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Store scraped articles
        self.articles = []
    
    def fetch_rss_feed(self, url):
        """Fetch articles from RSS feed"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
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
        """Scrape the full content of an article"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
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
            
            # Limit content length to avoid huge texts
            content = content[:5000] if len(content) > 5000 else content
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {str(e)}")
            return ""
    
    def scrape_news(self, min_articles=300):
        """Main scraping function to collect at least min_articles"""
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
                    
                    # Be respectful to servers
                    time.sleep(random.uniform(0.5, 1.5))
                
                if articles_collected >= min_articles:
                    break
            
            # If we haven't collected enough articles, try again
            if articles_collected < min_articles:
                logger.info(f"Still need {min_articles - articles_collected} more articles. Waiting before next round...")
                time.sleep(10)  # Wait before trying again
        
        logger.info(f"Completed scraping {len(self.articles)} articles")
        return self.articles
    
    def save_to_csv(self, filename='data/scraped_articles.csv'):
        """Save articles to CSV file"""
        df = pd.DataFrame(self.articles)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(self.articles)} articles to {filename}")
        return filename
    
    def save_to_sql(self, db_filename='data/scraped_articles.db'):
        """Save articles to SQLite database"""
        import sqlite3
        
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
    scraper = NewsScraper()
    
    # Scrape at least 300 articles
    articles = scraper.scrape_news(min_articles=300)
    
    # Save to CSV
    filename = scraper.save_to_csv()
    
    print(f"Successfully scraped and saved {len(articles)} articles to {filename}")


if __name__ == "__main__":
    main()