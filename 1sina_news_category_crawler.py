import aiohttp
import asyncio
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne
import datetime
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging
import chardet
import re
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json
from pathlib import Path
import requests
import argparse
import os
import pandas as pd
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置类
@dataclass
class Config:
    # MongoDB configuration
    MONGO_URI: str = "mongodb://localhost:27017/"
    DB_NAME: str = "sina_news_dataset_test"
    
    # Crawler configuration
    BASE_URL: str = "https://news.sina.com.cn/head/news{YYYYMMDD}{AMPM}.shtml"
    MAX_RETRIES_PER_URL: int = 3
    CONCURRENT_REQUESTS: int = 10  # Number of concurrent requests
    BATCH_SIZE: int = 100  # MongoDB batch write size
    
    # Date configuration
    START_DATE: str = "2025-01-01"  # Default start date
    END_DATE: str = "2025-06-30"    # Default end date
    EXCEL_FILE: str = "2025Q2_stock.xlsx"  # Excel file path
    DATE_COLUMN: str = "Date"       # Date column name
    
    # Category mapping
    CATEGORY_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        "headline": "Headlines",
        "news_item": "Headlines",
        "blk_cjxwgngjcj_01": "Domestic·International Finance",
        "blk_cjxwgpggmg_01": "Stocks·Hong Kong·US Stocks",
        "blk_cjxwlcsh_01": "Finance·Life",
        "blk_gnxw_01": "Domestic News",
        "blk_ndxw_01": "Domestic News",
        "blk_cjkjqcfc_01": "Finance·Tech·Auto·Real Estate",
        "blk_kjxwhlw_01": "Tech·Internet",
        "blk_kjxwkjts_01": "Tech·Exploration"
    })
    
    # News block configuration
    NEWS_BLOCKS: Dict[str, str] = field(default_factory=lambda: {
        "blk_yw_01": "Headlines",
        "blk_cjxwgngjcj_01": "Domestic·International Finance",
        "blk_cjxwgpggmg_01": "Stocks·Hong Kong·US Stocks",
        "blk_cjxwlcsh_01": "Finance·Life",
        "blk_gnxw_01": "Domestic News",
        "blk_ndxw_01": "Domestic News",
        "blk_cjkjqcfc_01": "Finance·Tech·Auto·Real Estate",
        "blk_kjxwhlw_01": "Tech·Internet",
        "blk_kjxwkjts_01": "Tech·Exploration"
    })

    # Failed URLs file path
    FAILED_URLS_FILE: str = "failed_urls.json"

    HEADERS: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # Encoding configuration
    ENCODINGS: List[str] = field(default_factory=lambda: [
        'gb18030',  # Prioritize GB18030 (backward compatible with GB2312 and GBK)
        'gbk',
        'gb2312',
        'utf-8'
    ])

    @classmethod
    def from_args(cls):
        """Create configuration from command line arguments"""
        parser = argparse.ArgumentParser(description='Sina News Crawler')
        
        # Add command line arguments
        parser.add_argument('--mongo-uri', type=str, help='MongoDB connection URI')
        parser.add_argument('--db-name', type=str, help='MongoDB database name')
        parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
        parser.add_argument('--excel-file', type=str, help='Path to Excel file containing date list')
        parser.add_argument('--date-column', type=str, help='Date column name in Excel file')
        parser.add_argument('--batch-size', type=int, help='Batch write size')
        parser.add_argument('--concurrent', type=int, help='Number of concurrent requests')
        
        args = parser.parse_args()
        
        # Create configuration instance
        config = cls()
        
        # Update configuration
        if args.mongo_uri:
            config.MONGO_URI = args.mongo_uri
        if args.db_name:
            config.DB_NAME = args.db_name
        if args.start_date:
            config.START_DATE = args.start_date
        if args.end_date:
            config.END_DATE = args.end_date
        if args.excel_file:
            config.EXCEL_FILE = args.excel_file
        if args.date_column:
            config.DATE_COLUMN = args.date_column
        if args.batch_size:
            config.BATCH_SIZE = args.batch_size
        if args.concurrent:
            config.CONCURRENT_REQUESTS = args.concurrent
            
        return config
    
    def get_date_list(self) -> List[datetime.date]:
        """Get list of dates to crawl"""
        # Parse start and end dates
        try:
            start_date = datetime.datetime.strptime(self.START_DATE, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(self.END_DATE, "%Y-%m-%d").date()
            logging.info(f"Date range set: {start_date} to {end_date}")
        except Exception as e:
            logging.error(f"Error parsing date range: {e}")
            sys.exit(1)
            
        # Read dates from Excel file
        if self.EXCEL_FILE and os.path.exists(self.EXCEL_FILE):
            try:
                logging.info(f"Reading date list from Excel file {self.EXCEL_FILE}")
                df = pd.read_excel(self.EXCEL_FILE)
                
                if self.DATE_COLUMN not in df.columns:
                    logging.error(f"Column '{self.DATE_COLUMN}' not found in Excel file")
                    logging.info(f"Available columns: {', '.join(df.columns)}")
                    sys.exit(1)
                
                # Extract date column and convert to datetime.date objects
                dates = []
                filtered_dates = []
                invalid_dates = 0
                
                for date_str in df[self.DATE_COLUMN]:
                    try:
                        # Try to convert various date formats
                        date_obj = None
                        
                        if isinstance(date_str, datetime.datetime) or isinstance(date_str, pd.Timestamp):
                            date_obj = date_str.date()
                        elif isinstance(date_str, str):
                            # Try parsing different date string formats
                            try:
                                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                            except ValueError:
                                try:
                                    date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d").date()
                                except ValueError:
                                    try:
                                        date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
                                    except ValueError:
                                        logging.warning(f"Cannot parse date: {date_str}, skipping")
                                        invalid_dates += 1
                                        continue
                        
                        if date_obj:
                            dates.append(date_obj)
                            # Only keep dates between start and end dates
                            if start_date <= date_obj <= end_date:
                                filtered_dates.append(date_obj)
                    except Exception as e:
                        logging.warning(f"Error processing date {date_str}: {e}")
                        invalid_dates += 1
                
                # Remove duplicates and sort
                unique_dates = sorted(list(set(filtered_dates)))
                
                logging.info(f"Read {len(dates)} dates from Excel file")
                logging.info(f"Found {len(unique_dates)} dates within specified range {start_date} to {end_date}")
                
                if invalid_dates > 0:
                    logging.warning(f"{invalid_dates} dates could not be parsed")
                
                if not unique_dates:
                    logging.error("No valid dates found in Excel file within specified range")
                    sys.exit(1)
                
                return unique_dates
                
            except Exception as e:
                logging.error(f"Error reading Excel file: {e}")
                sys.exit(1)
        else:
            logging.error(f"Excel file {self.EXCEL_FILE} not found")
            sys.exit(1)

class NewsSpider:
    def __init__(self, config: Config):
        self.config = config
        self.db = self._init_mongo()
        self.session = None
        self.failed_urls = []
        self.semaphore = asyncio.Semaphore(config.CONCURRENT_REQUESTS)

    def _init_mongo(self) -> MongoClient:
        """Initialize MongoDB connection"""
        client = MongoClient(self.config.MONGO_URI)
        return client[self.config.DB_NAME]

    async def _init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.config.HEADERS)

    def _detect_and_decode(self, content: bytes) -> str:
        """Smartly detect and decode content"""
        # List of encodings to prioritize
        encodings = self.config.ENCODINGS
        
        # First, use chardet to detect
        detected = chardet.detect(content)
        if detected and detected['confidence'] > 0.8:
            try:
                return content.decode(detected['encoding'])
            except (UnicodeDecodeError, LookupError):
                pass
        
        # If chardet detection fails, try other encodings in sequence
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all attempts fail, use errors='replace' for downgrade processing
        try:
            # Try gb18030 first as it's the most comprehensive Chinese encoding
            return content.decode('gb18030', errors='replace')
        except Exception:
            # Last resort: use utf-8 with replacement
            return content.decode('utf-8', errors='replace')

    async def fetch_url(self, url: str) -> Optional[str]:
        """Get URL content"""
        try:
            async with self.semaphore:
                for attempt in range(self.config.MAX_RETRIES_PER_URL):
                    try:
                        async with self.session.get(url, timeout=30) as response:
                            if response.status != 200:
                                if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                                    logging.warning(f"Failed to fetch URL: {url}, status code: {response.status}")
                                await asyncio.sleep(1)
                                continue
                            
                            content = await response.read()
                            return self._detect_and_decode(content)
                    except asyncio.TimeoutError:
                        if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                            logging.warning(f"Timeout fetching URL: {url}")
                        await asyncio.sleep(1)
                    except Exception as e:
                        if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                            logging.error(f"Failed to get URL {url}: {e}")
                        await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Failed to get URL {url}: {e}")
            self.failed_urls.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            })
            return None

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        
        # Remove special characters and control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Normalize whitespace characters
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove repeated punctuation symbols
        text = re.sub(r'[!?。！？，,]{2,}', lambda m: m.group()[0], text)
        # Remove empty parentheses
        text = re.sub(r'\(\s*\)|\[\s*\]|【\s*】|（\s*）', '', text)
        return text.strip()

    async def parse_news_list(self, html: str, date: datetime.date) -> List[Dict]:
        """Parse news list"""
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            news_list = []
            
            for block_id, category in self.config.NEWS_BLOCKS.items():
                block = soup.find('div', {'id': block_id})
                if not block:
                    continue
                    
                for link in block.find_all('a', href=True):
                    title = self._clean_text(link.get_text())
                    href = link['href']
                    
                    if not title or not href:
                        continue
                        
                    # Verify if title contains valid Chinese characters
                    if not re.search('[\u4e00-\u9fff]', title):
                        continue
                        
                    news_list.append({
                        'title': title,
                        'link': href,
                        'category': category,
                        'news_date': date.strftime('%Y-%m-%d'),
                        'fetch_date': datetime.datetime.now()
                    })
            
            return news_list
            
        except Exception as e:
            logging.error(f"Failed to parse news list: {e}")
            return []

    def _get_encoding(self, response: requests.Response) -> str:
        """Get correct encoding"""
        # Use chardet to detect encoding
        detected = chardet.detect(response.content)
        if detected and detected['confidence'] > 0.8:
            return detected['encoding']
        
        # If chardet detection fails or confidence is low, try to get from Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'charset=' in content_type:
            charset = content_type.split('charset=')[-1].strip()
            try:
                '测试'.encode(charset)  # Verify if encoding is valid
                return charset
            except:
                pass
        
        # If all else fails, return default encoding
        return 'gb18030'  # Use GB18030 (most comprehensive Chinese encoding)

    async def fetch_news(self, url: str, date_str: str) -> Optional[List[Dict]]:
        """Get news data"""
        try:
            for attempt in range(self.config.MAX_RETRIES_PER_URL):
                try:
                    async with self.session.get(url, timeout=30) as response:
                        if response.status != 200:
                            if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                                self._log_failed_url(url, date_str, 
                                                'am' if 'am' in url else 'pm', 
                                                f"HTTP error: {response.status}")
                            await asyncio.sleep(1)
                            continue
                        
                        # Use chardet to detect encoding
                        content = await response.read()
                        
                        # Try to decode with different encodings
                        text = None
                        for encoding in self.config.ENCODINGS:
                            try:
                                text = content.decode(encoding)
                                # Verify if decoded text contains Chinese characters
                                if re.search('[\u4e00-\u9fff]', text):
                                    break
                            except UnicodeDecodeError:
                                continue
                        
                        # If all encodings fail, use errors='replace'
                        if text is None:
                            text = content.decode('gb18030', errors='replace')
                            if not re.search('[\u4e00-\u9fff]', text):
                                text = content.decode('utf-8', errors='replace')
                        
                        # Verify decoded text
                        if not re.search('[\u4e00-\u9fff]', text):
                            if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                                logging.warning(f"Decoded text does not contain Chinese characters: {url}")
                                self._log_failed_url(url, date_str, 
                                                'am' if 'am' in url else 'pm', 
                                                "No Chinese characters in content")
                            await asyncio.sleep(1)
                            continue
                        
                        soup = BeautifulSoup(text, 'html.parser')
                        news_data = []

                        # Convert YYYYMMDD format to YYYY-MM-DD format
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

                        # Parse all news blocks
                        for block_id in self.config.NEWS_BLOCKS:
                            if block := soup.find('div', id=block_id):
                                block_news = self._parse_news_block(block, block_id, formatted_date)
                                if block_news:
                                    # Verify if title of each news contains valid Chinese characters
                                    valid_news = [
                                        news for news in block_news 
                                        if re.search('[\u4e00-\u9fff]', news.get('title', ''))
                                    ]
                                    news_data.extend(valid_news)
                                    
                                    if len(valid_news) < len(block_news):
                                        logging.warning(f"Found {len(block_news) - len(valid_news)} invalid titles")

                        await asyncio.sleep(random.uniform(1, 3))
                        return news_data
                
                except asyncio.TimeoutError:
                    if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                        self._log_failed_url(url, date_str, 
                                        'am' if 'am' in url else 'pm', 
                                        "Timeout")
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == self.config.MAX_RETRIES_PER_URL - 1:
                        logging.error(f"Failed to get URL {url}: {str(e)}")
                        self._log_failed_url(url, date_str, 
                                        'am' if 'am' in url else 'pm', 
                                        f"Request error: {str(e)}")
                    await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"Failed to get URL {url}: {str(e)}")
            self._log_failed_url(url, date_str, 
                             'am' if 'am' in url else 'pm', 
                             f"Request error: {str(e)}")
        
        return None

    def _parse_news_block(self, block: BeautifulSoup, block_id: str, date_str: str) -> List[Dict]:
        """Parse news block"""
        news_data = []
        
        try:
            # Handle different block types
            if block_id == "blk_yw_01":  # Headlines block
                for link in block.find_all('a', href=True):
                    title = self._clean_text(link.get_text())
                    if not title or not re.search('[\u4e00-\u9fff]', title):
                        continue
                        
                    news_data.append({
                        'title': title,
                        'link': link['href'],
                        'category': self.config.NEWS_BLOCKS[block_id],
                        'news_date': date_str,
                        'fetch_date': datetime.datetime.now()
                    })
            else:
                # Handle other blocks
                for link in block.find_all('a', href=True):
                    title = self._clean_text(link.get_text())
                    if not title or not re.search('[\u4e00-\u9fff]', title):
                        continue
                        
                    news_data.append({
                        'title': title,
                        'link': link['href'],
                        'category': self.config.NEWS_BLOCKS[block_id],
                        'news_date': date_str,
                        'fetch_date': datetime.datetime.now()
                    })
                    
        except Exception as e:
            logging.error(f"Failed to parse block {block_id}: {e}")
            
        return news_data

    def _log_failed_url(self, url: str, date: str, period: str, reason: str):
        """Record failed URL to JSON file"""
        failed_data = {
            "url": url,
            "date": date,
            "period": period,
            "reason": reason,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Read existing failed records
        try:
            if Path(self.config.FAILED_URLS_FILE).exists():
                with open(self.config.FAILED_URLS_FILE, 'r', encoding='utf-8') as f:
                    failed_urls = json.load(f)
            else:
                failed_urls = []
        except json.JSONDecodeError:
            failed_urls = []

        # Add new failed record
        failed_urls.append(failed_data)

        # Save updated records
        with open(self.config.FAILED_URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(failed_urls, f, ensure_ascii=False, indent=2)

        logging.info(f"Recorded failed URL: {url} ({date} {period})")

    async def process_date(self, date: datetime.date) -> List[Dict]:
        date_str = date.strftime("%Y%m%d")
        tasks = []
        
        for period in ["am", "pm"]:
            url = self.config.BASE_URL.format(YYYYMMDD=date_str, AMPM=period)
            for _ in range(self.config.MAX_RETRIES_PER_URL):
                news_data = await self.fetch_news(url, date_str)
                if news_data:
                    tasks.extend(news_data)
                    break
                await asyncio.sleep(random.uniform(1, 3))
        
        return tasks

    async def save_to_mongo(self, year: int, news_batch: List[Dict]):
        """Save news data to collection without suffix"""
        if not news_batch:
            return 0, 0, 0  # Return three zeros: total document count, new document count, duplicate document count

        # Use collection name without suffix
        collection = self.db[str(year)]  # For example: "2014" instead of "2014_1"
        
        # Get document count before saving
        before_count = collection.count_documents({})
        
        operations = [
            UpdateOne(
                {'link': news['link'], 'news_date': news['news_date']},  # Match both link and news_date
                {'$set': news},
                upsert=True
            ) for news in news_batch
        ]
        
        try:
            result = await asyncio.to_thread(
                collection.bulk_write,
                operations,
                ordered=False
            )
            
            # Get document count after saving
            after_count = collection.count_documents({})
            
            # Calculate new and duplicate document counts
            new_docs = after_count - before_count
            duplicate_docs = len(operations) - new_docs
            
            logging.info(f"Saved {len(operations)} news to {year} collection")
            logging.info(f"New documents: {new_docs} documents")
            logging.info(f"Duplicate documents: {duplicate_docs} documents")
            logging.info(f"Current collection document count: {after_count} documents")
            
            return after_count, new_docs, duplicate_docs
            
        except Exception as e:
            logging.error(f"Failed to save to MongoDB: {e}")
            return before_count, 0, 0

async def print_date_statistics(db: MongoClient):
    """Count documents for each date"""
    logging.info("\n=== News Date Statistics ===")
    
    # Iterate through all collections (years)
    for collection_name in db.list_collection_names():
        if not collection_name.isdigit():  # Skip non-year collections
            continue
            
        collection = db[collection_name]
        
        # Group by date and count
        pipeline = [
            {
                "$group": {
                    "_id": "$news_date",
                    "count": {"$sum": 1},
                    "categories": {"$addToSet": "$category"}
                }
            },
            {"$sort": {"_id": 1}}  # Sort by date
        ]
        
        results = collection.aggregate(pipeline)
        
        logging.info(f"\nYear: {collection_name}")
        logging.info("-" * 30)
        
        for result in results:
            date = result["_id"]
            count = result["count"]
            categories = result["categories"]
            
            logging.info(f"Date: {date}")
            logging.info(f"Total documents: {count}")
            logging.info(f"Contains categories: {', '.join(categories)}")
            logging.info("-" * 20)

async def clear_collections(db: MongoClient, start_year: int, end_year: int):
    """Clear all collections in specified year range (using collection without suffix)"""
    logging.info("\n=== Clear Collections ===")
    for year in range(start_year, end_year + 1):
        collection_name = str(year)  # Use collection name without suffix
        if collection_name in db.list_collection_names():
            result = db[collection_name].delete_many({})
            logging.info(f"Cleared {year} year collection, deleted {result.deleted_count} documents")
        else:
            logging.info(f"{year} year collection does not exist, no need to clear")
    logging.info("Collection clear completed\n")

async def main():
    # Create configuration from command line arguments
    config = Config.from_args()
    
    # Get list of dates to crawl
    date_list = config.get_date_list()
    
    if not date_list:
        logging.error("No dates found to crawl")
        return
    
    # Initialize crawler
    spider = NewsSpider(config)
    await spider._init_session()
    
    try:
        # Get all years in date range
        years = set(date.year for date in date_list)
        
        # Clear related collections
        await clear_collections(spider.db, min(years), max(years))
        
        news_batch = []
        current_year = None
        
        # Add document counter
        total_documents = 0
        processed_dates = 0
        
        # Iterate through date list
        for current_date in date_list:
            # Show progress
            processed_dates += 1
            logging.info(f"Processing date {current_date} ({processed_dates}/{len(date_list)})")
            
            # If year changes, save previous batch
            if current_year and current_year != current_date.year and news_batch:
                doc_count, new_docs, duplicate_docs = await spider.save_to_mongo(current_year, news_batch)
                total_documents = doc_count
                logging.info(f"Year changed, current database has {total_documents} documents")
                news_batch = []
            
            current_year = current_date.year
            date_news = await spider.process_date(current_date)
            
            if date_news:
                news_batch.extend(date_news)
                logging.info(f"Got {len(date_news)} news from {current_date}")
            else:
                logging.warning(f"No news got from {current_date}")
            
            # If batch reaches specified size, save to database
            if len(news_batch) >= config.BATCH_SIZE:
                doc_count, new_docs, duplicate_docs = await spider.save_to_mongo(current_year, news_batch)
                total_documents = doc_count
                logging.info(f"Batch saved, current database has {total_documents} documents")
                news_batch = []
            
            # Show progress
            logging.info(f"---------------------Completed processing date: {current_date}---------------------")
            
        # Save last batch data
        if news_batch:
            doc_count, new_docs, duplicate_docs = await spider.save_to_mongo(current_year, news_batch)
            total_documents = doc_count
            logging.info(f"Final saved, database has {total_documents} documents")
            
    except Exception as e:
        logging.error(f"Error occurred during processing: {e}")
    finally:
        if spider.session:
            await spider.session.close()
        
        # Save failed URL records
        if spider.failed_urls:
            spider.db.failed_urls.insert_many(spider.failed_urls)
            logging.info(f"Recorded {len(spider.failed_urls)} failed URLs")
        
        # Output document statistics for all years
        logging.info("\n=== All Year Document Statistics ===")
        for year in years:
            collection = spider.db[str(year)]
            doc_count = collection.count_documents({})
            logging.info(f"{year} year: {doc_count} documents")
        
        # Output date statistics
        await print_date_statistics(spider.db)

if __name__ == "__main__":
    asyncio.run(main())