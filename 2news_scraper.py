import os
import re
import time
import json
import random
import logging
import asyncio
import aiohttp
import requests
import datetime
import queue
import threading
import concurrent.futures
from bs4 import BeautifulSoup
from pymongo import MongoClient
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
from pathlib import Path
import argparse
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_scraper.log"),
        logging.StreamHandler()
    ]
)

# Global configuration
START_YEAR = 2025
END_YEAR = 2025
BATCH_SIZE = 100
MAX_WORKERS = 20
MAX_RETRIES = 3
TIMEOUT = 30
DELAY_RANGE = (0.5, 1.5)  # Random delay between requests in seconds
BATCH_DELAY = (1, 2)  # Random delay between batches in seconds
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sina_news_dataset_test"

# Parse command line arguments
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Scrape news content from URLs')
    parser.add_argument('--db-name', type=str, default=DB_NAME, help='MongoDB database name')
    parser.add_argument('--start-year', type=int, default=START_YEAR, help='Start year for processing')
    parser.add_argument('--end-year', type=int, default=END_YEAR, help='End year for processing')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Maximum number of worker threads')
    return parser.parse_args()

# Initialize MongoDB client
def init_mongo(db_name):
    """Initialize MongoDB connection"""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    return db

# Initialize session with retry mechanism
def get_session():
    """Get requests session with retry mechanism"""
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    })
    return session

# Normalize URL to avoid duplicates
def normalize_url(url: str) -> str:
    """Normalize URL to avoid duplicates"""
    # Remove query parameters and fragments
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    # Remove trailing slash
    if normalized.endswith('/'):
        normalized = normalized[:-1]
    
    return normalized.lower()

# Scrape news content
def scrape_news(url: str, original_title: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Scrape news content from URL"""
    try:
        session = get_session()
        response = session.get(url, timeout=TIMEOUT)
        
        if response.status_code != 200:
            logging.warning(f"Failed to fetch URL: {url}, status code: {response.status_code}")
            return None
        
        # Detect encoding
        if 'charset' in response.headers.get('Content-Type', '').lower():
            encoding = response.headers.get('Content-Type').split('charset=')[-1].strip()
        else:
            # Try to detect encoding from content
            encodings = ['utf-8', 'gb18030', 'gbk', 'gb2312']
            for enc in encodings:
                try:
                    response.content.decode(enc)
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            else:
                encoding = 'utf-8'  # Default to UTF-8 if detection fails
        
        # Parse HTML
        soup = BeautifulSoup(response.content.decode(encoding, errors='replace'), 'html.parser')
        
        # Extract title
        title = None
        if original_title:
            title = original_title
        else:
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.get_text().strip()
        
        if not title:
            logging.warning(f"No title found for URL: {url}")
            return None
        
        # Extract content
        content = ""
        article = soup.find('div', class_='article') or soup.find('div', id='artibody')
        if article:
            paragraphs = article.find_all('p')
            content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if not content:
            logging.warning(f"No content found for URL: {url}")
            return None
        
        # Extract publication date
        publish_date = None
        date_tag = soup.find('span', class_='date') or soup.find('span', class_='time-source')
        if date_tag:
            date_text = date_tag.get_text().strip()
            # Extract date using regex
            date_match = re.search(r'(\d{4})[-年](\d{1,2})[-月](\d{1,2})', date_text)
            if date_match:
                year, month, day = date_match.groups()
                publish_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # Extract image links
        image_links = []
        images = article.find_all('img') if article else []
        for img in images:
            src = img.get('src')
            if src:
                # Convert relative URLs to absolute
                abs_src = urljoin(url, src)
                image_links.append(abs_src)
        
        # Return structured data
        return {
            'title': title,
            'content': content,
            'url': url,
            'publish_date': publish_date,
            'fetch_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'image_links': image_links if image_links else "No images"
        }
        
    except Exception as e:
        logging.error(f"Error scraping URL: {url}, error: {str(e)}")
        return None

# Process news batch
def process_news_batch(news_batch: List[Dict], db, year: int, result_queue: queue.Queue, progress_bar=None):
    """Process a batch of news articles"""
    success_count = 0
    failed_urls = []
    
    # Create source and target collections
    source_collection = db[str(year)]
    target_collection = db[f"{year}_1"]
    
    # Process each news item
    for news in news_batch:
        try:
            # Check if already processed
            existing = target_collection.find_one({'url': news['link']})
            if existing:
                if progress_bar:
                    progress_bar.update(1)
                continue
            
            # Scrape news content
            scraped_data = scrape_news(news['link'], news.get('title'))
            
            if scraped_data:
                # Combine original news data with scraped content
                combined_data = {
                    'title': news.get('title', scraped_data['title']),
                    'category': news.get('category', ''),
                    'news_date': news.get('news_date', scraped_data['publish_date']),
                    'content': scraped_data['content'],
                    'url': news['link'],
                    'fetch_date': scraped_data['fetch_date'],
                    'image_links': scraped_data['image_links']
                }
                
                # Insert into target collection
                target_collection.insert_one(combined_data)
                success_count += 1
            else:
                failed_urls.append(news['link'])
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                
            # Add random delay to avoid overloading the server
            time.sleep(random.uniform(*DELAY_RANGE))
            
        except Exception as e:
            logging.error(f"Error processing news: {news.get('title', 'Unknown')}, URL: {news.get('link', 'Unknown')}, error: {str(e)}")
            failed_urls.append(news.get('link', 'Unknown'))
            if progress_bar:
                progress_bar.update(1)
    
    # Put results in queue
    result_queue.put((success_count, failed_urls))

# Update news in MongoDB
def update_news_in_mongo(db, news_data: List[Dict], year: int):
    """Update news content in MongoDB"""
    if not news_data:
        logging.warning(f"No news data for year {year}")
        return
    
    total = len(news_data)
    logging.info(f"Processing {total} news articles for year {year}")
    
    # Create target collection if not exists
    target_collection = db[f"{year}_1"]
    
    # Check existing records to avoid duplicates
    existing_urls = set()
    for doc in target_collection.find({}, {'url': 1}):
        if 'url' in doc:
            existing_urls.add(doc['url'])
    
    logging.info(f"Found {len(existing_urls)} existing records in target collection")
    
    # Filter out already processed URLs
    filtered_news = [news for news in news_data if news['link'] not in existing_urls]
    logging.info(f"After filtering, {len(filtered_news)} news articles need processing")
    
    if not filtered_news:
        logging.info(f"All news for {year} already processed")
        return
    
    # Process in batches using multiple threads
    result_queue = queue.Queue()
    success_count = 0
    all_failed_urls = []
    
    # Split into batches
    batches = [filtered_news[i:i+BATCH_SIZE] for i in range(0, len(filtered_news), BATCH_SIZE)]
    logging.info(f"Split into {len(batches)} batches of size {BATCH_SIZE}")
    
    # Create a single progress bar for all batches
    with tqdm(total=len(filtered_news), desc=f"Processing year {year}", unit="article") as progress_bar:
        # Process each batch
        for i, batch in enumerate(batches):
            logging.info(f"Processing batch {i+1}/{len(batches)}, size: {len(batch)}")
            
            # Create and start threads for this batch
            threads = []
            for j in range(0, len(batch), MAX_WORKERS):
                sub_batch = batch[j:j+MAX_WORKERS]
                thread = threading.Thread(
                    target=process_news_batch,
                    args=(sub_batch, db, year, result_queue, progress_bar)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Collect results
            while not result_queue.empty():
                batch_success, batch_failed = result_queue.get()
                success_count += batch_success
                all_failed_urls.extend(batch_failed)
                
            # Batch interval delay
            if i + 1 < len(batches):
                time.sleep(random.uniform(*BATCH_DELAY))
    
    # Clean up invalid data
    logging.info("Cleaning up invalid data...")
    target_collection.delete_many({"image_links": "No images"})
    target_collection.delete_many({"image_links": {"$exists": False}})
    target_collection.delete_many({"content": {"$exists": True, "$eq": ""}})
    
    # Print final statistics
    final_count = target_collection.count_documents({})
    logging.info(f"Year {year} processing complete: Total {total}, Success {success_count}, Final count {final_count}")

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Update global variables
    global DB_NAME, START_YEAR, END_YEAR, BATCH_SIZE, MAX_WORKERS
    DB_NAME = args.db_name
    START_YEAR = args.start_year
    END_YEAR = args.end_year
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.max_workers
    
    start_time = time.time()
    logging.info("=== Starting News Scraper ===")
    logging.info(f"Database: {DB_NAME}")
    logging.info(f"Year range: {START_YEAR} - {END_YEAR}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Max workers: {MAX_WORKERS}")
    
    # Initialize MongoDB
    db = init_mongo(DB_NAME)
    
    # Process news for each year
    for year in range(START_YEAR, END_YEAR + 1):
        logging.info(f"\nStarting to process news for year {year}")
        
        # Get news data
        news_data = list(db[str(year)].find())
        
        if news_data:
            logging.info(f"Found {len(news_data)} news articles for year {year}")
            update_news_in_mongo(db, news_data, year)
        else:
            logging.warning(f"No news data found for year {year}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    logging.info("=== Scraping Complete ===")

if __name__ == "__main__":
    main() 