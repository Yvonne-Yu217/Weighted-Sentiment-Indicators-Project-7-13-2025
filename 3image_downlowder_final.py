import os
import requests
from pymongo import MongoClient
from urllib.parse import urlparse
import time
import random
from datetime import datetime
import re
import json
import logging
from tqdm import tqdm
import warnings
import urllib3
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Set
import hashlib
import aiohttp
import asyncio
import aiodns
import platform
from pathlib import Path
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 禁用 SSL 警告
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 全局配置
START_YEAR = 2025
END_YEAR = 2025
MAX_WORKERS = 10  # 并行下载线程数
MAX_CONNECTIONS = 100  # 最大连接数
DOWNLOAD_TIMEOUT = 30  # 下载超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数
BATCH_SIZE = 200  # 每批处理的文档数
RATE_LIMIT_DELAY = (0.1, 0.5)  # 请求间隔随机延迟范围（秒）
SAVE_INTERVAL = 500  # 每处理多少条记录保存一次失败记录

# 用户代理列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67'
]

# 需要过滤的图片URL关键词
SKIP_KEYWORDS = [
    'icon', 
    'logo', 
    'p.v.iask', 
    'button', 
    'banner',
    'avatar',
    'loading',
    'background',
    'ad_',
    'adv_',
    'advertisement',
    'favicon',
    'footer',
    'header',
    'sidebar',
    'thumbnail',
    'placeholder'
]

class FailedDownloadTracker:
    """跟踪失败的下载记录"""
    
    def __init__(self, base_dir: str = None):
        """初始化失败下载追踪器"""
        self.failed_downloads = []
        self.downloaded_urls = set()  # 已下载的URL集合，避免重复下载
        self.base_dir = base_dir or os.getcwd()
        self.failed_file = os.path.join(self.base_dir, f"failed_downloads_{START_YEAR}-{END_YEAR}.json")
        self.lock = asyncio.Lock()  # 用于异步操作的锁
        
        # 加载已有的失败记录
        self._load_existing_failures()
    
    def _load_existing_failures(self):
        """加载已有的失败记录"""
        if os.path.exists(self.failed_file):
            try:
                with open(self.failed_file, 'r', encoding='utf-8') as f:
                    self.failed_downloads = json.load(f)
                    logging.info(f"加载了 {len(self.failed_downloads)} 条已有失败记录")
            except Exception as e:
                logging.error(f"加载失败记录出错: {e}")
    
    async def add_failed_download(self, doc_id: str, url: str, folder_path: str, error: str):
        """记录失败的下载"""
        async with self.lock:
            self.failed_downloads.append({
                "doc_id": str(doc_id),
                "url": url,
                "folder_path": folder_path,
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            })
    
    def add_downloaded_url(self, url: str):
        """添加已下载的URL"""
        self.downloaded_urls.add(url)
    
    def is_url_downloaded(self, url: str) -> bool:
        """检查URL是否已下载"""
        return url in self.downloaded_urls
    
    async def save_to_file(self):
        """保存失败记录到文件"""
        async with self.lock:
            if self.failed_downloads:
                os.makedirs(os.path.dirname(self.failed_file), exist_ok=True)
                with open(self.failed_file, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_downloads, f, ensure_ascii=False, indent=2)
                    logging.info(f"保存了 {len(self.failed_downloads)} 条失败记录到 {self.failed_file}")

class ImageDownloader:
    """图片下载器"""
    
    def __init__(self, db, tracker: FailedDownloadTracker):
        """初始化下载器"""
        self.db = db
        self.tracker = tracker
        self.base_dir = os.path.join(os.getcwd(), 'images')
        self.year_dir = None
        self.session = None
        self.connector = None
        self.processed_count = 0
        self.success_count = 0
        self.skip_count = 0
        self.error_count = 0
        self.current_year = None
        
        # 创建基础目录
        os.makedirs(self.base_dir, exist_ok=True)
    
    async def init_session(self):
        """初始化异步HTTP会话"""
        # 根据操作系统选择DNS解析器
        if platform.system() == 'Windows':
            # Windows下使用默认解析器
            resolver = None
        else:
            # Linux/Mac下使用aiodns
            resolver = aiohttp.AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])
        
        # 创建TCP连接器
        self.connector = aiohttp.TCPConnector(
            limit=MAX_CONNECTIONS,
            ssl=False,
            use_dns_cache=True,
            ttl_dns_cache=300,
            resolver=resolver
        )
        
        # 创建客户端会话
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT),
            headers={
                'Accept': 'image/webp,image/*,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Connection': 'keep-alive'
            }
        )
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    def is_valid_image_url(self, url: str) -> bool:
        """检查图片URL是否有效"""
        if not url or not isinstance(url, str):
            return False
            
        url_lower = url.lower()
        
        # 检查URL格式
        if not (url_lower.startswith('http://') or url_lower.startswith('https://')):
            return False
            
        # 检查文件扩展名
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        has_valid_extension = any(url_lower.endswith(ext) for ext in valid_extensions)
        
        # 如果没有有效扩展名，检查URL路径是否包含图片相关字符串
        if not has_valid_extension:
            path = urlparse(url_lower).path
            if not any(ext in path for ext in valid_extensions):
                return False
        
        # 检查是否包含需要跳过的关键词
        return not any(keyword in url_lower for keyword in SKIP_KEYWORDS)
    
    def get_image_filename(self, url: str) -> str:
        """从URL获取图片文件名，如果无法获取则生成一个唯一的文件名"""
        # 尝试从URL获取文件名
        image_name = os.path.basename(urlparse(url).path)
        
        # 如果文件名为空或无效，生成一个基于URL的哈希文件名
        if not image_name or len(image_name) < 5 or '.' not in image_name:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            # 尝试从URL中提取文件扩展名
            ext = os.path.splitext(urlparse(url).path)[1]
            if not ext or ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                ext = '.jpg'  # 默认使用jpg扩展名
            image_name = f"image_{url_hash}{ext}"
        
        return image_name
    
    async def download_image(self, url: str, folder_path: str, doc_id: str) -> bool:
        """异步下载单张图片，带重试机制"""
        # 检查URL是否已下载
        if self.tracker.is_url_downloaded(url):
            self.skip_count += 1
            return True
            
        # 检查URL是否有效
        if not self.is_valid_image_url(url):
            await self.tracker.add_failed_download(
                doc_id=doc_id,
                url=url,
                folder_path=folder_path,
                error="Invalid image URL"
            )
            self.skip_count += 1
            return False
        
        for attempt in range(MAX_RETRIES):
            try:
                # 随机选择用户代理
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                
                async with self.session.get(url.strip(), headers=headers) as response:
                    if response.status == 200:
                        # 检查内容类型
                        content_type = response.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            await self.tracker.add_failed_download(
                                doc_id=doc_id,
                                url=url,
                                folder_path=folder_path,
                                error=f"Not an image: {content_type}"
                            )
                            return False
                        
                        # 获取文件名
                        image_name = self.get_image_filename(url)
                        file_path = os.path.join(folder_path, image_name)
                        
                        # 确保目录存在
                        os.makedirs(folder_path, exist_ok=True)
                        
                        # 下载图片
                        image_data = await response.read()
                        
                        # 检查图片大小
                        if len(image_data) < 1024:  # 小于1KB的图片可能是无效的
                            await self.tracker.add_failed_download(
                                doc_id=doc_id,
                                url=url,
                                folder_path=folder_path,
                                error="Image too small (< 1KB)"
                            )
                            return False
                        
                        # 保存图片
                        with open(file_path, 'wb') as f:
                            f.write(image_data)
                        
                        # 记录已下载的URL
                        self.tracker.add_downloaded_url(url)
                        self.success_count += 1
                        return True
                    else:
                        # 如果是最后一次尝试，记录失败
                        if attempt == MAX_RETRIES - 1:
                            await self.tracker.add_failed_download(
                                doc_id=doc_id,
                                url=url,
                                folder_path=folder_path,
                                error=f"HTTP {response.status}"
                            )
                
                # 重试前等待
                await asyncio.sleep(random.uniform(1, 3))
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    await self.tracker.add_failed_download(
                        doc_id=doc_id,
                        url=url,
                        folder_path=folder_path,
                        error=str(e)
                    )
                    self.error_count += 1
                else:
                    await asyncio.sleep(random.uniform(2, 5))
        
        return False
    
    def extract_date_from_news(self, news: Dict) -> str:
        """从新闻数据中提取日期"""
        # 尝试从news_date字段获取
        news_date = news.get('news_date')
        if news_date:
            # 如果是日期对象，转换为字符串
            if isinstance(news_date, datetime):
                return news_date.strftime('%Y%m%d')
            # 如果是字符串，尝试标准化格式
            elif isinstance(news_date, str):
                # 尝试解析各种可能的日期格式
                date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y年%m月%d日']
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(news_date, fmt)
                        return parsed_date.strftime('%Y%m%d')
                    except ValueError:
                        continue
                
                # 如果无法解析，尝试从字符串中提取日期
                match = re.search(r'(\d{4})[-/.]?(\d{1,2})[-/.]?(\d{1,2})', news_date)
                if match:
                    year, month, day = match.groups()
                    return f"{year}{month:0>2}{day:0>2}"
        
        # 尝试从URL中提取日期
        link = news.get('link', '')
        if link:
            match = re.search(r'/(\d{4}[-/]\d{2}[-/]\d{2})/', link)
            if match:
                date_str = match.group(1)
                # 移除分隔符
                return re.sub(r'[-/]', '', date_str)
        
        # If cannot extract date, use current date
        return datetime.now().strftime('%Y%m%d')
    
    def get_safe_title(self, title: str, max_length: int = 100) -> str:
        """Get safe title as folder name"""
        if not title:
            return f"untitled_{int(time.time())}"
            
        # Remove unsafe characters
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        
        # 限制长度
        if len(safe_title) > max_length:
            safe_title = safe_title[:max_length]
            
        return safe_title
    
    async def process_news(self, news: Dict) -> None:
        """处理单条新闻数据"""
        doc_id = news.get('_id')
        title = news.get('title', '')
        image_links = news.get('image_links', '')
        
        if not image_links or image_links == "No images":
            return
            
        try:
            # 直接使用数据库中的news_date字段，如果存在的话
            news_date = news.get('news_date', '')
            if not news_date:
                news_date = self.extract_date_from_news(news)
            
            # 确保news_date是字符串格式
            if isinstance(news_date, datetime):
                news_date = news_date.strftime('%Y%m%d')
                
            folder_path_date = os.path.join(self.year_dir, str(news_date))
            
            # 创建文章标题文件夹
            safe_title = self.get_safe_title(title)
            folder_path_article = os.path.join(folder_path_date, safe_title)
            
            # 解析图片URL列表
            image_urls = []
            if isinstance(image_links, str):
                # 如果是逗号分隔的字符串
                image_urls = [url.strip() for url in image_links.split(",") 
                             if url.strip() and url.strip() != "No images"]
            elif isinstance(image_links, list):
                # 如果已经是列表
                image_urls = [url for url in image_links if url]
                
            # 下载图片
            download_tasks = []
            downloaded_images = []  # 存储成功下载的图片路径
            
            for url in image_urls:
                # 添加随机延迟，避免请求过于频繁
                await asyncio.sleep(random.uniform(*RATE_LIMIT_DELAY))
                
                # 获取图片文件名
                image_name = self.get_image_filename(url)
                image_path = os.path.join(folder_path_article, image_name)
                
                # 添加下载任务和路径
                download_tasks.append(self.download_image(url, folder_path_article, doc_id))
                downloaded_images.append(image_path)
                
            # 并行下载图片
            if download_tasks:
                results = await asyncio.gather(*download_tasks)
                
                # 如果有图片成功下载，则记录映射关系
                if any(results):
                    # 创建映射记录
                    mapping = {
                        "original_id": doc_id,
                        "title": title,
                        "news_date": news_date,
                        "folder_path": folder_path_article,
                        "image_paths": downloaded_images,
                        "link": news.get('link', ''),
                        "has_images": True,
                        "year": self.current_year,
                        "processed": False  # 标记是否已处理
                    }
                    
                    # 存储到映射集合
                    await self.save_mapping(mapping)
                
        except Exception as e:
            logging.error(f"处理新闻出错 {doc_id}: {e}")
            await self.tracker.add_failed_download(
                doc_id=doc_id,
                url="article_processing_failed",
                folder_path="",
                error=str(e)
            )
            self.error_count += 1
            
    async def save_mapping(self, mapping: Dict) -> None:
        """保存图片与新闻的映射关系"""
        try:
            # 使用 {year}_mapping 集合存储映射关系
            mapping_collection = self.db[f"{self.current_year}_mapping"]
            
            # 使用asyncio.to_thread将同步操作转为异步
            def insert_mapping():
                mapping_collection.insert_one(mapping)
                
            await asyncio.to_thread(insert_mapping)
            logging.info(f"保存映射关系: {mapping['original_id']}")
        except Exception as e:
            logging.error(f"保存映射关系失败: {e}")
    
    async def process_batch(self, news_batch: List[Dict]) -> None:
        """处理一批新闻数据"""
        tasks = []
        for news in news_batch:
            tasks.append(self.process_news(news))
            
        await asyncio.gather(*tasks)
        
        # 更新处理计数
        self.processed_count += len(news_batch)
        
        # 定期保存失败记录
        if self.processed_count % SAVE_INTERVAL == 0:
            await self.tracker.save_to_file()
            logging.info(f"已处理: {self.processed_count}, 成功: {self.success_count}, "
                        f"跳过: {self.skip_count}, 错误: {self.error_count}")
    
    async def download_all_images(self, year: int) -> None:
        """下载所有新闻图片"""
        self.year_dir = os.path.join(self.base_dir, f"{year}_1")
        os.makedirs(self.year_dir, exist_ok=True)
        self.current_year = year
        
        collection_name = f"{year}_1"
        collection = self.db[collection_name]
        
        # 确保映射集合存在
        mapping_collection_name = f"{year}_mapping"
        if mapping_collection_name not in self.db.list_collection_names():
            self.db.create_collection(mapping_collection_name)
        
        # 初始化HTTP会话
        await self.init_session()
        
        try:
            # 获取总文档数
            total_docs = collection.count_documents({"image_links": {"$exists": True}})
            logging.info(f"处理集合 {collection_name}, 共 {total_docs} 条记录")
            
            processed = 0
            with tqdm(total=total_docs, desc=f"下载进度") as pbar:
                while processed < total_docs:
                    try:
                        # 获取一批数据
                        news_batch = list(collection.find(
                            {"image_links": {"$exists": True}},
                            skip=processed,
                            limit=BATCH_SIZE
                            
                        ))
                        
                        if not news_batch:
                            break
                            
                        # 处理这批数据
                        await self.process_batch(news_batch)
                        
                        # 更新进度条
                        pbar.update(len(news_batch))
                        processed += len(news_batch)
                        
                    except Exception as e:
                        logging.error(f"处理批次时出错: {e}")
                        await asyncio.sleep(5)  # 出错时等待一段时间再继续
                        continue
                        
        finally:
            # 保存最终的失败记录
            await self.tracker.save_to_file()
            
            # 关闭HTTP会话
            await self.close_session()
            
            # 打印最终统计
            logging.info(f"\n下载完成! 总处理: {self.processed_count}, 成功: {self.success_count}, "
                        f"跳过: {self.skip_count}, 错误: {self.error_count}")

async def retry_failed_downloads(db, year: int):
    """重试失败的下载（优化版）"""
    failed_file = os.path.join(os.getcwd(), 'images', f"failed_downloads_{START_YEAR}-{END_YEAR}.json")
    
    if not os.path.exists(failed_file):
        logging.info("没有找到失败记录文件")
        return
        
    with open(failed_file, 'r', encoding='utf-8') as f:
        failed_records = json.load(f)
    
    if not failed_records:
        logging.info("没有需要重试的下载")
        return
    
    # 过滤掉文章处理失败的记录，只保留图片URL下载失败的记录
    valid_records = [r for r in failed_records if r['url'] != "article_processing_failed"]
    
    if not valid_records:
        logging.info("没有有效的失败下载记录需要重试")
        return
        
    logging.info(f"开始重试 {len(valid_records)} 个失败的下载")
    
    # 创建下载器和追踪器
    tracker = FailedDownloadTracker(os.path.join(os.getcwd(), 'images'))
    downloader = ImageDownloader(db, tracker)
    
    # 初始化HTTP会话，增加并发连接数
    downloader.connector = aiohttp.TCPConnector(
        limit=MAX_CONNECTIONS * 2,  # 增加连接数
        ssl=False,
        use_dns_cache=True,
        ttl_dns_cache=300
    )
    
    downloader.session = aiohttp.ClientSession(
        connector=downloader.connector,
        timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT),
        headers={
            'Accept': 'image/webp,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }
    )
    
    try:
        # 批量处理，每批次并发下载多个图片
        batch_size = 50  # 每批处理50个
        new_failures = []
        
        for i in range(0, len(valid_records), batch_size):
            batch = valid_records[i:i+batch_size]
            tasks = []
            
            for record in batch:
                # 确保文件夹存在
                os.makedirs(record['folder_path'], exist_ok=True)
                
                # 创建下载任务
                tasks.append(downloader.download_image(
                    record['url'], 
                    record['folder_path'],
                    record['doc_id']
                ))
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(results):
                record = batch[j]
                if isinstance(result, Exception) or result is False:
                    new_failures.append(record)
            
            # 更新进度
            logging.info(f"已重试 {min(i + batch_size, len(valid_records))}/{len(valid_records)} 个下载")
        
        # 保存新的失败记录
        if new_failures:
            retry_file = os.path.join(os.getcwd(), 'images', f"failed_downloads_retry_{START_YEAR}-{END_YEAR}.json")
            with open(retry_file, 'w', encoding='utf-8') as f:
                json.dump(new_failures, f, ensure_ascii=False, indent=2)
            logging.info(f"仍有 {len(new_failures)} 个下载失败，已保存到 {retry_file}")
        else:
            logging.info("所有失败的下载都已成功重试!")
            
    finally:
        # 关闭HTTP会话
        await downloader.session.close()
        await downloader.connector.close()

# 工具函数：判断文件夹下是否有图片文件
def has_image_files(folder):
    """检查文件夹中是否包含图片文件"""
    if not os.path.exists(folder):
        return False
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            return True
    return False

# 下载主流程最后增加清理空title层文件夹并更新数据库
async def clean_empty_title_folders_and_update_db(db, year):
    """清理没有图片的标题文件夹并更新数据库"""
    logging.info(f"开始清理 {year} 年空图片文件夹...")
    year_dir = os.path.join(os.getcwd(), 'images', f"{year}_1")
    mapping_collection = db[f"{year}_mapping"]
    news_collection = db[f"{year}_1"]
    removed_count = 0
    nonexistent_count = 0
    
    # 处理不存在的文件夹路径
    for mapping in mapping_collection.find({"has_images": True}):
        folder_path = mapping.get("folder_path")
        if not os.path.exists(folder_path):
            # 更新映射记录
            mapping_collection.update_one(
                {"_id": mapping["_id"]},
                {"$set": {"has_images": False, "processed": True}}
            )
            
            # 更新原始新闻记录
            if "original_id" in mapping:
                news_collection.update_one(
                    {"_id": mapping["original_id"]},
                    {"$set": {"has_images": False}}
                )
            
            nonexistent_count += 1
    
    if nonexistent_count > 0:
        logging.info(f"已处理 {nonexistent_count} 个不存在的文件夹路径")
    
    # 处理存在但为空的文件夹
    if not os.path.exists(year_dir):
        logging.warning(f"{year}年图片目录不存在: {year_dir}")
        return
        
    for date_folder in os.listdir(year_dir):
        date_path = os.path.join(year_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
            
        for title_folder in os.listdir(date_path):
            title_path = os.path.join(date_path, title_folder)
            if not os.path.isdir(title_path):
                continue
                
            if not has_image_files(title_path):
                # 删除空title文件夹
                try:
                    shutil.rmtree(title_path)
                    logging.info(f"已删除空图片文件夹: {title_path}")
                except Exception as e:
                    logging.warning(f"删除文件夹失败: {title_path}, 错误: {e}")
                    
                # 查找对应的映射记录
                mappings = list(mapping_collection.find({"folder_path": title_path}))
                for mapping in mappings:
                    # 更新映射记录
                    mapping_collection.update_one(
                        {"_id": mapping["_id"]}, 
                        {"$set": {"has_images": False}}
                    )
                    
                    # 如果有原始ID，同时更新新闻记录
                    if "original_id" in mapping:
                        news_collection.update_one(
                            {"_id": mapping["original_id"]},
                            {"$set": {"has_images": False}}
                        )
                
                removed_count += 1
                
    logging.info(f"{year}年清理空title文件夹并更新数据库完成，共处理 {removed_count} 个空文件夹，{nonexistent_count} 个不存在的文件夹路径。")

def delete_dot_underscore_files(root_dir):
    """递归删除目录下所有._开头的文件"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("._"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"已删除隐藏文件: {file_path}")
                except Exception as e:
                    logging.warning(f"删除隐藏文件失败: {file_path}, 错误: {e}")

async def main():
    """主函数"""
    start_time = time.time()
    logging.info(f"开始下载 {START_YEAR}-{END_YEAR} 年的图片...")
    
    # 初始化 MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['sina_news_dataset_test']
    
    # 检查集合是否存在
    for year in range(START_YEAR, END_YEAR + 1):
        collection_name = f"{year}_1"
        if collection_name not in db.list_collection_names():
            logging.error(f"集合 {collection_name} 不存在，请先创建并导入数据")
            continue
        
        # ======= 自动化重置数据库和图片目录 =======
        # 删除mapping集合
        mapping_collection_name = f"{year}_mapping"
        if mapping_collection_name in db.list_collection_names():
            db[mapping_collection_name].drop()
            logging.info(f"已删除集合 {mapping_collection_name}")
        
        # 重置原始新闻集合中的图片标记
        original_collection = db[f"{year}_1"]
        original_collection.update_many(
            {}, 
            {"$set": {"has_images": False}}
        )
        logging.info(f"已重置 {year}_1 集合中的图片标记")
        
        # 删除图片目录前先清理所有._隐藏文件
        images_base_dir = os.path.join(os.getcwd(), 'images')
        year_dir = os.path.join(images_base_dir, f"{year}_1")
        if os.path.exists(year_dir):
            delete_dot_underscore_files(year_dir)
            shutil.rmtree(year_dir)
            logging.info(f"已删除图片目录 {year_dir}")
        # ======= END =======
        
        # 初始化失败追踪器
        tracker = FailedDownloadTracker()
        downloader = ImageDownloader(db, tracker)
        try:
            await downloader.download_all_images(year=year)
            
            # 直接重试失败的下载
            logging.info("开始重试失败的下载...")
            await retry_failed_downloads(db, year=year)
            
            # 下载结束后清理空title层文件夹并同步数据库
            await clean_empty_title_folders_and_update_db(db, year)
            
        except KeyboardInterrupt:
            logging.info("用户中断下载")
            await tracker.save_to_file()
        except Exception as e:
            logging.error(f"程序出错: {e}")
            await tracker.save_to_file()
    
    elapsed_time = time.time() - start_time
    logging.info(f"下载完成! 总耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    # 在Windows上需要使用不同的事件循环策略
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行主函数
    asyncio.run(main()) 