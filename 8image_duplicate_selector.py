import os
import cv2
import numpy as np
import logging
import time
import uuid
from tqdm import tqdm
from pymongo import MongoClient
from typing import Dict, Any, List
from PIL import Image as PILImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_duplicate_selector.log', encoding='utf-8')
    ]
)

# Read years to process from environment variable
years_env = os.environ.get('YEARS_TO_PROCESS')
if years_env:
    years_to_process = [int(y) for y in years_env.split(',')]
    logging.info(f"Loaded years to process from environment variable: {years_to_process}")
else:
    years_to_process = list(range(2025, 2026))  # Default: process 2014-2024
    logging.info(f"Using default years to process: {years_to_process}")

# Global configuration
HASH_SIZE = 16  # 感知哈希大小
SIMILARITY_THRESHOLD = 0.75  # 相似度阈值，越高要求越严格（调低了阈值以检测更多可能的重复）
DEBUG_MODE = True  # 调试模式，输出更多日志信息
RESIZE_SIZE = 16  # 调整为偏数大小，OpenCV的DCT函数要求输入的尺寸必须是偶数

class ImageDuplicateSelector:
    """Image Duplicate Detection and Best Quality Selector"""
    
    def __init__(self):
        """Initialize selector"""
        # Statistics
        self.total_images = 0
        self.duplicate_images = 0
        self.duplicate_groups = 0
        self.unique_images = 0
        self.news_processed = 0
        self.found_image_count = 0     # Number of images found successfully
        self.missing_image_count = 0   # Number of missing images
        self.success_hash_count = 0    # Number of images with hash computed successfully
        self.failed_hash_count = 0     # Number of images failed to compute hash
        
        # Detailed missing images info
        self.missing_images_detail = []
        
        # Image path cache for faster lookup
        self.path_cache = {}
        # Cache of mapping collection folder paths: {original_id -> folder_path}
        self.folder_path_cache = {}
        # Cache of successful path patterns for statistics
        self.successful_patterns = {}
    
    def compute_phash(self, image_path: str) -> np.ndarray:
        try:
            # 使用PIL读取图片，支持更多格式
            try:
                with PILImage.open(image_path) as pil_img:
                    # 如果是GIF等动画格式，只取第一帧
                    if getattr(pil_img, "is_animated", False):
                        pil_img.seek(0)  # 确保是第一帧
                    
                    # 转换为RGB模式（处理RGBA、索引色等）
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                        
                    # 调整大小并转为灰度
                    pil_img = pil_img.resize((RESIZE_SIZE, RESIZE_SIZE), PILImage.LANCZOS)
                    pil_img = pil_img.convert('L')  # 转为灰度
                    
                    # 转为numpy数组
                    img = np.array(pil_img)
                    
                    # 记录图片尺寸和增加成功计算哈希的计数器
                    self.success_hash_count += 1
                    if DEBUG_MODE and (self.success_hash_count % 50 == 0):
                        logging.info(f"成功计算哈希: {self.success_hash_count} 张, 当前图片: {image_path}, 尺寸={img.shape}")
            except Exception as pil_error:
                # PIL读取失败，尝试使用OpenCV
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"无法使用OpenCV读取图片: {image_path}")
                    
                # 记录图片尺寸和增加成功计算哈希的计数器
                self.success_hash_count += 1
                if DEBUG_MODE and (self.success_hash_count % 50 == 0):
                    logging.info(f"成功计算哈希(OpenCV): {self.success_hash_count} 张, 当前图片: {image_path}, 尺寸={img.shape}")
                    
                # 调整图片大小，转换为灰度
                img = cv2.resize(img, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            # 计算DCT
            try:
                dct = cv2.dct(np.float32(img))
                # 取左上角较低频率区域，确保使用HASH_SIZE大小
                if HASH_SIZE < RESIZE_SIZE:
                    dct_low = dct[:HASH_SIZE, :HASH_SIZE]
                else:
                    dct_low = dct
            except Exception as e:
                # 捕获DCT计算错误，尝试使用非DCT的方法
                if DEBUG_MODE:
                    logging.warning(f"DCT计算失败，使用备用方法: {e}")
                # 使用更简单的平均哈希来替代DCT
                img_resized = cv2.resize(img, (HASH_SIZE, HASH_SIZE)) if len(img.shape) > 2 else img
                # 将图像作为低频率部分
                dct_low = img_resized
            
            # 计算均值
            med = np.median(dct_low)
            
            # 得到哈希值
            phash = dct_low > med
            
            return phash
            
        except Exception as e:
            # 增加哈希计算失败计数器
            self.failed_hash_count += 1
            if self.failed_hash_count % 10 == 0 or DEBUG_MODE:
                logging.warning(f"哈希计算失败: {self.failed_hash_count} 张, 当前图片: {image_path}: {e}")
            return None
    
    def calculate_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """计算两个哈希值之间的相似度"""
        if hash1 is None or hash2 is None:
            if DEBUG_MODE:
                logging.warning("无法计算相似度: 至少一个哈希值为None")
            return 0.0
        
        # 计算哈希的汉明距离
        distance = np.count_nonzero(hash1 != hash2)
        
        # 转换为相似度 (0-1范围，1为完全相同)
        max_distance = HASH_SIZE * HASH_SIZE
        similarity = 1.0 - (distance / max_distance)
        
        # 在调试模式下记录高相似度图片
        if DEBUG_MODE and similarity > 0.6:  # 记录所有相似度超过 0.6 的对比
            logging.info(f"发现相似图片: 相似度={similarity:.4f}, 距离={distance}")
        
        return similarity
    
    def process_news_images(self, news_doc: Dict[str, Any], mapping_collection=None, year=None) -> Dict[str, Any]:
        """处理单个新闻文档中的所有图片，检测重复并选择最佳质量"""
        # 如果没有all_images字段或者它不是列表，直接返回原始文档
        if "all_images" not in news_doc or not isinstance(news_doc["all_images"], list):
            return news_doc

        all_images = news_doc["all_images"]
        if len(all_images) <= 1:
            # 只有一张或没有图片，不需要去重
            # 但仍然添加标记字段
            for img in all_images:
                img["duplicate_group_id"] = None
                img["use_for_sentiment"] = True
            return news_doc

        # 统计更新
        self.total_images += len(all_images)
        
        # 创建图片字典用于比较
        images_dict = {}
        base_image_dir = "/Volumes/storage/Quantifying Investor Sentiment in the Chinese Stock Market through News Media/images/"
        processed_image_dir = "/Volumes/storage/Quantifying Investor Sentiment in the Chinese Stock Market through News Media/processed_images/"
        
        # 遍历图片，计算感知哈希
        news_date = news_doc.get("news_date", "")
        news_title = news_doc.get("title", "")
        doc_year = news_date.split("-")[0] if news_date and "-" in news_date else (year or "")
        news_id = news_doc.get("original_id", "")
        
        if not news_date or not news_title or not doc_year:
            # 缺少日期或标题信息，无法构建路径
            logging.warning(f"无法构建图片路径: 缺少日期或标题信息 news_id={news_id}")
            for img in all_images:
                img["duplicate_group_id"] = None
                img["use_for_sentiment"] = True
            return news_doc
        
        # 获取新闻的原始ID，用于查找folder_path
        original_id = news_doc.get("original_id")
        img_folder_path = None
        # 将 ObjectId 转为字符串，便于后续字符串比较
        news_id_str = str(news_id) if news_id else ""
        
        # 从mapping集合获取folder_path
        if original_id and mapping_collection is not None:
            # 先从缓存中查询
            if original_id in self.folder_path_cache:
                img_folder_path = self.folder_path_cache[original_id]
                if DEBUG_MODE:
                    logging.debug(f"从缓存获取folder_path: {img_folder_path}")
            else:
                # 从mapping集合查询
                try:
                    mapping_doc = mapping_collection.find_one({"original_id": original_id})
                    if mapping_doc and "folder_path" in mapping_doc:
                        img_folder_path = mapping_doc["folder_path"]
                        # 缓存folder_path以提高性能
                        self.folder_path_cache[original_id] = img_folder_path
                        if DEBUG_MODE:
                            logging.debug(f"从映射集合获取folder_path: {img_folder_path}")
                except Exception as e:
                    logging.warning(f"从映射集合获取folder_path失败: {e}")
        
        # 为每张图片计算感知哈希
        for i, img in enumerate(all_images):
            rel_path = img.get("path")
            if not rel_path:
                continue
                
            # 提取文件名，用于构建备用路径
            file_name = os.path.basename(rel_path)
            file_base, file_ext = os.path.splitext(file_name)
                
            # 初始化变量
            image_path = None
            possible_paths = []
            actual_path = img.get('actual_path') 
            img_path = img.get('image_path')
            
            # 首先检查缓存中是否有此图片路径
            img_key = f"{news_doc.get('_id')}_{rel_path}"
            if img_key in self.path_cache:
                image_path = self.path_cache[img_key]
                if os.path.exists(image_path):
                    self.found_image_count += 1
                    if DEBUG_MODE and (self.found_image_count % 100 == 0):
                        logging.info(f"从缓存中找到图片，总共找到 {self.found_image_count} 张图片")
                else:
                    # 移除无效缓存
                    del self.path_cache[img_key]
                    image_path = None
            
            # 如果没有从缓存找到有效路径，则构建新路径
            if image_path is None:
                # 优先方式1: 使用actual_path（如果存在）
                if actual_path:
                    possible_paths.append(actual_path)
                    if DEBUG_MODE:
                        logging.debug(f"尝试路径(actual_path): {actual_path}")
                # 方式2: 使用folder_path + path构建绝对路径
                if img_folder_path:
                    direct_path = os.path.join(img_folder_path, rel_path)
                    possible_paths.append(direct_path)
                    if DEBUG_MODE:
                        logging.debug(f"尝试路径(folder_path): {direct_path}")
                

                
                # 方式3: 使用图片记录中的image_path
                if img_path:
                    possible_paths.append(img_path)
                    if not os.path.isabs(img_path):
                        abs_img_path = os.path.abspath(img_path)
                        possible_paths.append(abs_img_path)
                # 备用路径构建方式 - 基于年份的目录结构
                # 路径格式4: 尝试在年份_1/日期/标题目录下查找
                path4 = os.path.join(base_image_dir, f"{doc_year}_1", news_date, news_title, rel_path)
                possible_paths.append(path4)
                
                # 路径格式5: 尝试在年份_1/日期目录下查找
                path5 = os.path.join(base_image_dir, f"{doc_year}_1", news_date, rel_path)
                possible_paths.append(path5)
                
                # 路径格式6: 尝试在年份_1/日期/新闻ID目录下查找
                if news_id:
                    path6 = os.path.join(base_image_dir, f"{doc_year}_1", news_date, str(news_id), rel_path)
                    possible_paths.append(path6)
                    
                # 路径格式7: 尝试在年份目录下直接查找（不考虑日期和标题）
                path7 = os.path.join(base_image_dir, f"{doc_year}_1", file_name)
                possible_paths.append(path7)
                
                # 路径格式8: 尝试在processed_images/年份目录下直接查找
                path8 = os.path.join(processed_image_dir, f"{doc_year}_1", file_name)
                possible_paths.append(path8)
                path9 = os.path.join(processed_image_dir, f"{doc_year}_2", file_name)
                possible_paths.append(path9)
                
                # 路径格式10: 尝试在images目录下直接查找
                path11 = os.path.join(base_image_dir, file_name)
                possible_paths.append(path11)
            
            # 检查路径是否存在
            image_path = None
            pattern_used = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    # 增加成功找到的图片计数器
                    self.found_image_count += 1
                    
                    # 记录成功的路径模式
                    if os.path.isabs(rel_path) and path == rel_path:
                        pattern_used = "absolute"
                    elif path.startswith(base_image_dir):
                        if news_title in path and news_date in path:
                            pattern_used = "year_date_title"
                        elif news_date in path and news_id_str and news_id_str in path:
                            pattern_used = "year_date_id"
                        elif news_date in path:
                            pattern_used = "year_date"
                        elif os.path.basename(path) == file_name:
                            pattern_used = "direct_filename"
                        else:
                            # 检查是否使用了不同扩展名
                            for ext in [".jpg", ".png", ".gif", ".jpeg", ".webp"]:
                                if path.endswith(f"{file_base}{ext}"):
                                    pattern_used = f"alt_ext_{ext[1:]}"
                                    break
                    elif path.startswith(processed_image_dir):
                        if f"{year}_2" in path:
                            pattern_used = "processed_year2"
                        elif f"{year}_1" in path:
                            pattern_used = "processed_year1"
                        elif os.path.basename(path) == file_name:
                            pattern_used = "processed_direct_filename"
                    
                    # 更新成功模式计数
                    if pattern_used:
                        self.successful_patterns[pattern_used] = self.successful_patterns.get(pattern_used, 0) + 1
                    
                    # 将成功的路径添加到缓存，使用新的键格式
                    img_key = f"{news_doc.get('_id')}_{rel_path}"
                    self.path_cache[img_key] = path
                    
                    # 记录成功的路径构建方式
                    if path.startswith(img_folder_path) if img_folder_path else False:
                        pattern_used = "direct_folder_path"
                    
                    if DEBUG_MODE and (self.found_image_count % 100 == 0):
                        logging.info(f"总共找到 {self.found_image_count} 张图片，使用模式: {pattern_used}")
                        # 每500张图片，输出成功模式统计
                        if self.found_image_count % 500 == 0:
                            patterns_sorted = sorted(self.successful_patterns.items(), key=lambda x: x[1], reverse=True)
                            logging.info(f"成功路径模式统计: {patterns_sorted[:5]}")
                    break
            
            # 如果找不到图片，记录日志并继续
            if not image_path:
                # 增加缺失图片计数器
                self.missing_image_count += 1
                
                # 在图片信息中标记缺失
                img["missing"] = True
                img["duplicate_group_id"] = None
                img["use_for_sentiment"] = False  # 缺失图片不用于情感分析
                
                reason = "File does not exist"
                if not rel_path:
                    reason = "Empty path"
                elif not os.path.isabs(rel_path):
                    reason = "Not absolute path"
                self.missing_images_detail.append({
                    "news_id": news_doc.get("original_id"),
                    "img_path": rel_path,
                    "reason": reason
                })
                if i == 0 or (self.missing_image_count % 100 == 0):  # 控制日志输出频率
                    logging.warning(f"找不到图片 (第 {self.missing_image_count} 张缺失): {rel_path} | NewsID: {news_doc.get('original_id')} | Reason: {reason}")
                continue
            
            # 图片存在，计算哈希
            phash = self.compute_phash(image_path)
            if phash is not None:
                # 在图片信息中记录实际路径，便于后续处理
                img["actual_path"] = image_path
                img["missing"] = False
                
                images_dict[i] = {
                    "index": i,
                    "phash": phash,
                    "path": image_path,
                    "info": img
                }
            else:
                # 无法计算哈希的图片也标记为不用于情感分析
                img["duplicate_group_id"] = None
                img["use_for_sentiment"] = False
                img["hash_failed"] = True
        
        # 检测重复图片
        duplicate_groups = {}
        processed_indices = set()
        
        # 比较所有图片对
        for i in images_dict.keys():
            if i in processed_indices:
                continue
                
            # 创建新的组
            group = [images_dict[i]]
            processed_indices.add(i)
            
            # 与其他图片比较
            for j in images_dict.keys():
                if j in processed_indices or i == j:
                    continue
                    
                # 计算相似度
                similarity = self.calculate_similarity(
                    images_dict[i]["phash"],
                    images_dict[j]["phash"]
                )
                
                # 如果相似度超过阈值，加入组
                if similarity >= SIMILARITY_THRESHOLD:
                    group.append(images_dict[j])
                    processed_indices.add(j)
            
            # 如果组内有超过1张图片，则是重复组
            if len(group) > 1:
                # 生成组ID
                group_id = str(uuid.uuid4())
                duplicate_groups[group_id] = group
        
        # 更新统计信息
        self.duplicate_groups += len(duplicate_groups)
        duplicate_count = 0
        for group in duplicate_groups.values():
            duplicate_count += len(group) - 1
        self.duplicate_images += duplicate_count
        
        # 标记所有图片
        for img in all_images:
            # 默认设置为非重复、可用于情感分析
            img["duplicate_group_id"] = None
            img["use_for_sentiment"] = True
        
        # 处理重复组
        for group_id, group in duplicate_groups.items():
            # 按quality_score排序选择最佳图片
            sorted_group = sorted(group, key=lambda x: x["info"].get("quality_score", 0), reverse=True)
            best_index = sorted_group[0]["index"]
            
            # 标记组内所有图片
            for item in group:
                index = item["index"]
                all_images[index]["duplicate_group_id"] = group_id
                # 只有最佳图片用于情感分析
                all_images[index]["use_for_sentiment"] = (index == best_index)
        
        # 统计处理结果
        dupe_count = 0
        use_count = 0
        for img in all_images:
            if img.get("duplicate_group_id") is not None:
                dupe_count += 1
            if img.get("use_for_sentiment", False):
                use_count += 1
        
        if dupe_count > 0:
            logging.info(f"新闻ID: {news_doc.get('original_id')} - 发现 {dupe_count} 张重复图片，{use_count} 张用于情感分析")
        
        return news_doc
    
    def process_year_images(self, year, db):
        """处理指定年份的图片，检测重复并选择最佳质量"""
        # 获取要处理的集合
        source_collection_name = f"{year}_2"  # 质量评分后的集合
        target_collection_name = f"{year}_sentiment"  # 用于情感分析的集合
        mapping_collection_name = f"{year}_mapping"  # 原始映射集合，包含folder_path
        
        try:
            collections = db.list_collection_names()

            # 1. 检查源集合是否存在
            if source_collection_name not in collections:
                logging.warning(f"Source collection '{source_collection_name}' not found, skipping year {year}.")
                return

            # 2. 根据用户要求，在处理前重置目标集合
            if target_collection_name in collections:
                logging.info(f"Resetting target collection: Dropping '{target_collection_name}'...")
                db.drop_collection(target_collection_name)
                logging.info(f"Collection '{target_collection_name}' dropped.")

            # 3. 获取集合对象
            source_collection = db[source_collection_name]
            target_collection = db[target_collection_name]  # 将在首次插入时自动创建

            # 4. 检查并获取 mapping 集合
            mapping_collection = None
            if mapping_collection_name in collections:
                mapping_collection = db[mapping_collection_name]
                logging.info(f"Found mapping collection '{mapping_collection_name}' for path resolution.")
            else:
                logging.warning(f"Mapping collection '{mapping_collection_name}' not found. Path resolution may be limited.")

            # 准备处理
            doc_count = source_collection.count_documents({})
            if doc_count == 0:
                logging.info(f"Source collection '{source_collection_name}' is empty. Skipping year {year}.")
                return
            logging.info(f"Source collection '{source_collection_name}' contains {doc_count} documents to process.")
            
            # 按新闻ID分组处理图片
            news_ids = source_collection.distinct("original_id")
            logging.info(f"发现 {len(news_ids)} 个不同的新闻ID")
            
            # 先检查第一条数据的结构
            if news_ids:
                sample_news_id = news_ids[0]
                first_record = source_collection.find_one({"original_id": sample_news_id})
                if first_record:
                    logging.info(f"第一条记录字段: {', '.join(list(first_record.keys()))}")
                    
                    # 输出关键信息
                    news_title = first_record.get('title', '[无标题]')
                    news_date = first_record.get('news_date', '[无日期]')
                    news_id = first_record.get('original_id', '[无ID]')
                    logging.info(f"示例新闻: ID={news_id}, 日期={news_date}, 标题={news_title}")
                    
                    # 检查图片字段
                    if "all_images" in first_record:
                        img_list = first_record['all_images']
                        if isinstance(img_list, list):
                            img_count = len(img_list)
                            logging.info(f"all_images图片数量: {img_count}")
                            
                            # 输出第一张图片详细信息
                            if img_count > 0:
                                first_img = img_list[0]
                                logging.info(f"第一张图片详细信息:\n{first_img}")
                                
                                # 特别注意质量分数字段
                                has_quality_score = 'quality_score' in first_img
                                logging.info(f"是否包含质量分数字段: {has_quality_score}")
                                
                                # 测试查找其中一张图片
                                self.test_image_path_construction(first_record)
                        else:
                            logging.warning(f"all_images字段不是列表类型: {type(img_list)}")
                    else:
                        logging.warning(f"记录中没有 all_images 字段")
            
            # 随机抛查70个记录往目标集合插入试试，看是否有权限问题
            if news_ids and len(news_ids) > 70:
                test_ids = news_ids[:10]  # 只取10个测试
                test_docs = list(source_collection.find({"original_id": {"$in": test_ids}}))
                if test_docs:
                    try:
                        # 尝试插入测试数据
                        result = target_collection.insert_many(test_docs)
                        logging.info(f"成功插入 {len(result.inserted_ids)} 条测试数据")
                        # 成功后清空回来
                        target_collection.delete_many({"original_id": {"$in": test_ids}})
                    except Exception as e:
                        logging.error(f"测试插入数据失败: {e}")
            
            # 记录处理统计
            processed_news_count = 0
            processed_image_count = 0
            duplicate_image_count = 0
            duplicate_group_count = 0
            success_hash_count = 0  # 成功生成哈希的图片数量
            failed_hash_count = 0   # 无法生成哈希的图片数量
            found_image_count = 0    # 成功找到的图片数量
            missing_image_count = 0   # 丢失的图片数量
            
            # 设置每一批处理的数量
            batch_size = 100
            total_batches = (len(news_ids) + batch_size - 1) // batch_size
            
            # 分批处理新闻，避免内存问题
            with tqdm(total=len(news_ids), desc=f"处理 {year} 年的新闻") as pbar:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(news_ids))
                    batch_ids = news_ids[start_idx:end_idx]
                    
                    # 处理当前批次的新闻
                    batch_docs = []
                    for news_id in batch_ids:
                        news_doc = source_collection.find_one({"original_id": news_id})
                        if not news_doc:
                            continue
                        
                        # 记录当前重复组数量和重复图片数量，用于后续统计
                        before_groups = self.duplicate_groups
                        before_duplicates = self.duplicate_images
                        before_found_images = found_image_count
                        before_missing_images = missing_image_count
                        
                        try:
                            # 处理新闻文档中的图片，检测重复，传入mapping_collection用于构建路径
                            processed_doc = self.process_news_images(news_doc, mapping_collection, year)
                            
                            # 计算统计数据
                            if processed_doc and "all_images" in processed_doc and isinstance(processed_doc["all_images"], list):
                                img_count = len(processed_doc["all_images"])
                                processed_image_count += img_count
                                
                                # 计算该新闻新增的信息
                                new_groups = self.duplicate_groups - before_groups
                                new_duplicates = self.duplicate_images - before_duplicates
                                new_found = found_image_count - before_found_images
                                new_missing = missing_image_count - before_missing_images
                                
                                duplicate_group_count += new_groups
                                duplicate_image_count += new_duplicates
                                
                                # 如果有重复图片，输出详细信息
                                if new_groups > 0:
                                    news_title = processed_doc.get('title', '[无标题]')
                                    logging.info(f"[重复图片] 新闻 '{news_title}' 包含 {new_groups} 组重复图片, 共 {new_duplicates} 张")
                            
                            # 将处理后的文档添加到批次数据中
                            batch_docs.append(processed_doc)
                            processed_news_count += 1
                            
                        except Exception as e:
                            logging.error(f"处理新闻ID {news_id} 出错: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # 更新进度条
                        pbar.update(1)
                    
                    # 批量写入到数据库
                    if batch_docs:
                        try:
                            result = target_collection.insert_many(batch_docs)
                            logging.info(f"批次 {batch_idx+1}/{total_batches}: 写入 {len(result.inserted_ids)} 条数据到 {target_collection_name}")
                        except Exception as e:
                            logging.error(f"写入数据库时出错: {e}")
            
            # 输出处理结果统计
            logging.info(f"\n===== 年份 {year} 处理统计 =====")
            logging.info(f"- 处理新闻数: {processed_news_count}")
            logging.info(f"- 处理图片数: {processed_image_count}")
            logging.info(f"- 找到图片数: {self.found_image_count} ({self.found_image_count/processed_image_count*100 if processed_image_count else 0:.2f}% 找到率)")
            logging.info(f"- 缺失图片数: {self.missing_image_count} ({self.missing_image_count/processed_image_count*100 if processed_image_count else 0:.2f}% 缺失率)")
            logging.info(f"- 成功计算哈希数: {self.success_hash_count} ({self.success_hash_count/self.found_image_count*100 if self.found_image_count else 0:.2f}% 成功率)")
            logging.info(f"- 哈希计算失败数: {self.failed_hash_count} ({self.failed_hash_count/self.found_image_count*100 if self.found_image_count else 0:.2f}% 失败率)")
            logging.info(f"- 发现重复组数: {duplicate_group_count}")
            logging.info(f"- 发现重复图片数: {duplicate_image_count}")
            
            # 计算去重率
            if processed_image_count > 0:
                dedup_rate = (duplicate_image_count / processed_image_count) * 100
                logging.info(f"- 去重率: {dedup_rate:.2f}%")
            
            self.news_processed += processed_news_count
            
        except Exception as e:
            logging.error(f"处理 {year} 年图片时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def test_image_path_construction(self, sample_record):
        """测试图片路径构建，检查是否可以正确访问文件"""
        if not sample_record or "all_images" not in sample_record or not isinstance(sample_record["all_images"], list) or not sample_record["all_images"]:
            logging.warning("样本记录无效，无法测试图片路径")
            return
            
        # 获取第一张图片信息
        first_img = sample_record["all_images"][0]
        rel_path = first_img.get("path")
        if not rel_path:
            logging.warning("图片路径为空")
            return
            
        # 基础图片目录
        base_image_dir = "/Volumes/storage/Quantifying Investor Sentiment in the Chinese Stock Market through News Media/images/"
        
        # 获取新闻信息
        news_date = sample_record.get("news_date", "")
        news_title = sample_record.get("title", "")
        year = news_date.split("-")[0] if news_date and "-" in news_date else ""
        
        if not news_date or not news_title or not year:
            logging.warning(f"缺少必要信息: 日期={news_date}, 标题={news_title}, 年份={year}")
            return

        # 构建可能的路径
        possible_paths = []
        
        # 路径格式1: 年份_1/日期/新闻标题/图片名
        path1 = os.path.join(base_image_dir, f"{year}_1", news_date, news_title, rel_path)
        possible_paths.append(path1)
        
        # 路径格式2: 直接使用相对路径（如果是绝对路径）
        if os.path.isabs(rel_path):
            possible_paths.append(rel_path)
        
        # 检查所有可能的路径
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                logging.info(f"找到图片: {path}")
                break
        
        if not found_path:
            logging.warning(f"所有尝试的路径均不存在: {possible_paths}")
            
            # 尝试查找基础目录下是否有其他文件
            year_dir = os.path.join(base_image_dir, f"{year}_1")
            if os.path.exists(year_dir):
                try:
                    date_dirs = os.listdir(year_dir)
                    logging.info(f"年份目录 {year_dir} 包含 {len(date_dirs)} 个日期目录")
                    if date_dirs and news_date in date_dirs:
                        news_date_dir = os.path.join(year_dir, news_date)
                        try:
                            title_dirs = os.listdir(news_date_dir)
                            logging.info(f"日期目录 {news_date_dir} 包含 {len(title_dirs)} 个新闻标题目录")
                            # 列出前5个标题示例
                            if title_dirs:
                                logging.info(f"标题示例: {title_dirs[:5]}")
                        except Exception as e:
                            logging.error(f"无法列出日期目录内容: {e}")
                except Exception as e:
                    logging.error(f"无法列出年份目录内容: {e}")
            else:
                logging.warning(f"年份目录不存在: {year_dir}")
        
        return found_path is not None
    
    def print_summary(self):
        """打印处理统计摘要"""
        logging.info("\n========== 图片重复检测和选择总结 ==========\n")
        
        # 数据量统计
        logging.info("1. 数据量统计:")
        logging.info(f"- 处理新闻总数: {self.news_processed:,} 篇")
        logging.info(f"- 处理图片总数: {self.total_images:,} 张")
        
        # 图片检索和处理情况
        logging.info("\n2. 图片检索情况:")
        found_image_percent = (self.found_image_count / self.total_images * 100) if self.total_images > 0 else 0
        missing_image_percent = (self.missing_image_count / self.total_images * 100) if self.total_images > 0 else 0
        logging.info(f"- 找到图片数量: {self.found_image_count:,} 张 ({found_image_percent:.2f}%)")
        logging.info(f"- 缺失图片数量: {self.missing_image_count:,} 张 ({missing_image_percent:.2f}%)")
        
        # 哈希计算情况
        logging.info("\n3. 哈希计算情况:")
        success_hash_percent = (self.success_hash_count / self.found_image_count * 100) if self.found_image_count > 0 else 0
        failed_hash_percent = (self.failed_hash_count / self.found_image_count * 100) if self.found_image_count > 0 else 0
        logging.info(f"- 哈希计算成功: {self.success_hash_count:,} 张 ({success_hash_percent:.2f}%)")
        logging.info(f"- 哈希计算失败: {self.failed_hash_count:,} 张 ({failed_hash_percent:.2f}%)")
        
        # 重复检测结果
        logging.info("\n4. 重复检测结果:")
        # 计算重复图片组内的总图片数量
        total_imgs_in_groups = self.duplicate_groups + self.duplicate_images # 主图片 + 重复图片
        avg_img_per_group = (total_imgs_in_groups / self.duplicate_groups) if self.duplicate_groups > 0 else 0
        
        logging.info(f"- 发现重复图片组: {self.duplicate_groups:,} 组")
        logging.info(f"- 重复组内总图片数量: {total_imgs_in_groups:,} 张")
        logging.info(f"- 平均每组图片数: {avg_img_per_group:.2f} 张/组")
        logging.info(f"- 保留代表图片: {self.duplicate_groups:,} 张 (每组保留最高质量的一张)")
        logging.info(f"- 被排除的重复图片: {self.duplicate_images:,} 张")
        logging.info(f"- 非重复图片数量: {self.total_images - total_imgs_in_groups:,} 张")
        
        # 效率分析
        logging.info("\n5. 效率分析:")
        if self.total_images > 0:
            # 计算被排除图片的比例
            excluded_rate = (self.duplicate_images / self.total_images * 100)
            # 计算重复组内图片占比
            group_imgs_rate = (total_imgs_in_groups / self.total_images * 100) if self.total_images > 0 else 0
            # 计算最终用于情感分析的图片比例 (非重复 + 重复组代表图片)
            usable_ratio = 100 - excluded_rate
            
            logging.info(f"- 重复组图片占比: {group_imgs_rate:.2f}% (包含代表图片和被排除图片)")
            logging.info(f"- 被排除图片占比: {excluded_rate:.2f}%")
            logging.info(f"- 用于情感分析的图片占比: {usable_ratio:.2f}% (非重复图片+每组代表图片)")
            
            if self.duplicate_images > 0:
                efficiency_gain = self.duplicate_images / self.total_images * 100
                logging.info(f"- 去重减少情感计算量: {efficiency_gain:.2f}% (不必对重复图片重复计算)")
                if self.duplicate_groups > 0 and self.duplicate_images > 0:
                    duplicate_reduction_ratio = (self.duplicate_images / (self.duplicate_groups + self.duplicate_images)) * 100
                    logging.info(f"- 重复组内去重效率: {duplicate_reduction_ratio:.2f}% (每组只保留一张代表图片)")
        
        # 缺失图片统计
        if self.total_images > 0:
            missing_ratio = self.missing_image_count / self.total_images * 100
            logging.info(f"总共处理图片数: {self.total_images}")
            logging.info(f"缺失图片数: {self.missing_image_count} ({missing_ratio:.2f}%)")
            if self.missing_image_count > 0:
                logging.info("部分缺失图片详细信息:")
                for detail in self.missing_images_detail[:10]:
                    logging.info(f"新闻ID: {detail['news_id']}, 图片路径: {detail['img_path']}, 原因: {detail['reason']}")

def main():
    """Main function"""
    start_time = time.time()
    logging.info("=== Starting image duplicate detection and selection ===")
    
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['sina_news_dataset_test']
    
    # Create image duplicate selector
    selector = ImageDuplicateSelector()
    
    # Process images for each year
    for year in years_to_process:
        selector.process_year_images(year, db)
    
    # Output statistics
    logging.info(f"Processing complete! Total images processed: {selector.total_images}")
    logging.info(f"Duplicate groups found: {selector.duplicate_groups}, Total duplicate images: {selector.duplicate_images}")
    logging.info(f"Hashes computed successfully: {selector.success_hash_count}, Failed: {selector.failed_hash_count}")
    logging.info(f"Images found: {selector.found_image_count}, Missing: {selector.missing_image_count}")
    
    # Output successful path pattern statistics
    if selector.successful_patterns:
        patterns_sorted = sorted(selector.successful_patterns.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"Successful path pattern statistics:")
        for pattern, count in patterns_sorted:
            logging.info(f"  - {pattern}: {count} images")
        
        # Output path cache size
        logging.info(f"Path cache size: {len(selector.path_cache)} records")
    
    # Print total summary
    selector.print_summary()
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    logging.info("=== Image duplicate detection and selection finished ===")

if __name__ == "__main__":
    main()
