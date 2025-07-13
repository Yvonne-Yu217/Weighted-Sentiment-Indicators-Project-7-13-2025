import os
import cv2
import numpy as np
import logging
import time
from PIL import Image
from tqdm import tqdm
from pymongo import MongoClient
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 从环境变量读取年份配置
years_env = os.environ.get('YEARS_TO_PROCESS')
if years_env:
    years_to_process = [int(y) for y in years_env.split(',')]
    logging.info(f"从环境变量读取年份配置: {years_to_process}")
else:
    years_to_process = list(range(2025, 2026))  # 默认处理2014-2024年
    logging.info(f"使用默认年份配置: {years_to_process}")

# 全局配置
ASPECT_RATIO_THRESHOLD = 3.0  # 宽高比阈值
MIN_IMAGE_SIZE = 150  # 最小图片尺寸
MAX_IMAGE_SIZE = 4000  # 最大图片尺寸
MIN_CONTENT_AREA = 0.15  # 最小内容区域占比

class ImageBasicFilter:
    """图片基础筛选器：处理无效图片、尺寸比例、二维码检测和内容区域"""
    
    def __init__(self):
        """初始化筛选器"""
        # 初始化QR码检测器
        self.qr_detector = cv2.QRCodeDetector()
        
        # 统计信息
        self.total_images = 0
        self.invalid_images = 0
        self.size_ratio_rejected = 0
        self.qr_code_images = 0
        self.content_area_rejected = 0
        self.passed_images = 0
    
    def is_valid_image(self, image_path: str) -> tuple:
        """检查图片是否有效并返回PIL Image对象"""
        try:
            # 尝试打开图片
            img = Image.open(image_path)
            
            # 检查图片是否可以正常读取
            img.verify()
            
            # 重新打开图片（verify后需要重新打开）
            img = Image.open(image_path)
            
            # 检查是否为空图片
            if img.size[0] == 0 or img.size[1] == 0:
                return False, None
                
            return True, img
            
        except Exception as e:
            logging.debug(f"无效图片 {image_path}: {e}")
            return False, None
    
    def check_image_size(self, img: Image.Image) -> bool:
        """检查图片尺寸是否合适"""
        width, height = img.size
        
        # 检查图片是否太小
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False
            
        # 检查图片是否太大
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            return False
            
        # 检查宽高比是否合适
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
            return False
            
        return True
    
    def check_content_area(self, cv_image: np.ndarray) -> bool:
        """检查图片内容区域是否足够大"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 使用简单的全局二值化代替自适应阈值，速度更快
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 计算内容区域占比
            content_ratio = np.count_nonzero(binary) / (binary.shape[0] * binary.shape[1])
            
            # 判断内容区域是否足够大
            return content_ratio >= MIN_CONTENT_AREA
            
        except Exception:
            # 处理异常情况，默认通过
            return True
    
    def has_qr_code(self, image_path: str) -> bool:
        """检测图片是否包含二维码"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            # 检测二维码
            data, bbox, _ = self.qr_detector.detectAndDecode(img)
            
            # 如果检测到二维码
            if bbox is not None and len(bbox) > 0:
                return True
                
            return False
            
        except Exception:
            return False
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """处理单个图片，进行基础筛选"""
        self.total_images += 1
        
        # 检查图片有效性
        is_valid, pil_image = self.is_valid_image(image_path)
        if not is_valid:
            self.invalid_images += 1
            return {"valid": False, "reject_reason": "invalid_image"}
        
        try:
            # 检查图片尺寸和比例
            if not self.check_image_size(pil_image):
                self.size_ratio_rejected += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "size_ratio"}
            
            # 转换为OpenCV格式用于内容区域检查
            cv_image = cv2.imread(image_path)
            
            # 检查内容区域
            if not self.check_content_area(cv_image):
                self.content_area_rejected += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "content_area"}
            
            # 检查是否包含二维码
            if self.has_qr_code(image_path):
                self.qr_code_images += 1
                pil_image.close()
                return {"valid": False, "reject_reason": "qr_code"}
            
            # 通过所有基础筛选
            self.passed_images += 1
            pil_image.close()
            return {"valid": True, "path": image_path}
            
        except Exception as e:
            logging.error(f"处理图片出错 {image_path}: {e}")
            if pil_image:
                pil_image.close()
            self.invalid_images += 1
            return {"valid": False, "reject_reason": "error", "error": str(e)}
    
    def process_year_images(self, year, db):
        """处理指定年份的图片，进行基础筛选"""
        mapping_collection_name = f"{year}_mapping"
        
        # 检查映射集合是否存在
        if mapping_collection_name not in db.list_collection_names():
            logging.warning(f"映射集合 {mapping_collection_name} 不存在，跳过处理")
            return
        
        # 获取映射集合
        mapping_collection = db[mapping_collection_name]
        
        # 创建或获取中间集合（存储通过基础筛选的图片）
        filtered_collection_name = f"{year}_filtered"
        filtered_collection = db[filtered_collection_name]
        
        # 查询未处理且有图片的映射记录
        unprocessed_mappings = mapping_collection.find({"processed": False, "has_images": True})
        total_mappings = mapping_collection.count_documents({"processed": False, "has_images": True})
        
        if total_mappings == 0:
            logging.info(f"{year} 年没有未处理的映射记录")
            return
        
        logging.info(f"开始基础筛选 {year} 年的 {total_mappings} 条未处理映射记录")
        
        # 使用进度条
        with tqdm(total=total_mappings, desc=f"基础筛选 {year} 年图片") as pbar:
            # 遍历每条映射记录
            for mapping in unprocessed_mappings:
                try:
                    original_id = mapping.get("original_id")
                    folder_path = mapping.get("folder_path")
                    
                    # 检查路径是否存在
                    if not os.path.isabs(folder_path):
                        abs_folder_path = os.path.abspath(folder_path)
                    else:
                        abs_folder_path = folder_path
                    
                    if os.path.exists(abs_folder_path):
                        valid_images = []
                        rejected_images = []
                        
                        # 遍历文件夹中的所有图片
                        for root, _, files in os.walk(abs_folder_path):
                            for file in files:
                                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                                    image_path = os.path.join(root, file)
                                    
                                    # 计算相对路径，用于存储
                                    rel_path = os.path.relpath(image_path, abs_folder_path)
                                    
                                    try:
                                        # 处理图片
                                        result = self.process_single_image(image_path)
                                        
                                        if result["valid"]:
                                            # 将通过筛选的图片添加到有效列表
                                            valid_images.append({
                                                "path": rel_path,
                                                "abs_path": image_path
                                            })
                                        else:
                                            # 记录被拒绝的图片
                                            rejected_images.append({
                                                "path": rel_path,
                                                "reason": result.get("reject_reason", "unknown")
                                            })
                                    except Exception as e:
                                        logging.error(f"处理图片出错 {image_path}: {e}")
                        
                        # 从原始集合获取完整的新闻内容
                        source_collection = db[f"{year}_1"]
                        news_content = source_collection.find_one({"_id": original_id})
                        
                        if news_content:
                            # 创建新的文档，包含通过筛选的图片信息
                            new_doc = {
                                "original_id": original_id,
                                "title": news_content.get("title", ""),
                                "content": news_content.get("content", ""),
                                "news_date": news_content.get("news_date", ""),
                                "category": news_content.get("category", ""),
                                "link": mapping.get("link", ""),  # 从mapping获取link，而不是从news_content
                                "valid_images": valid_images,
                                "rejected_images": rejected_images,
                                "has_valid_images": len(valid_images) > 0,
                                "basic_filtered": True,
                                "quality_processed": False  # 标记为未进行质量处理
                            }
                            
                            # 插入或更新到过滤集合
                            filtered_collection.update_one(
                                {"original_id": original_id},
                                {"$set": new_doc},
                                upsert=True
                            )
                            
                            # 更新映射记录的基础筛选状态
                            mapping_collection.update_one(
                                {"_id": mapping["_id"]},
                                {"$set": {"basic_filtered": True}}
                            )
                    else:
                        logging.warning(f"文件夹不存在: {folder_path}")
                
                except Exception as e:
                    logging.error(f"处理映射记录出错: {e}")
                
                # 更新进度条
                pbar.update(1)
    
    def print_summary(self):
        """打印处理统计摘要"""
        logging.info("\n=== 基础筛选统计 ===")
        logging.info(f"总图片数量: {self.total_images}")
        logging.info(f"通过筛选: {self.passed_images}")
        
        def percent(count):
            return f"{(count / self.total_images * 100):.2f}%" if self.total_images > 0 else "0.00%"
        
        logging.info("\n被筛选的图片:")
        logging.info(f"- 无效图片: {self.invalid_images} ({percent(self.invalid_images)})")
        logging.info(f"- 尺寸/比例不符: {self.size_ratio_rejected} ({percent(self.size_ratio_rejected)})")
        logging.info(f"- 二维码图片: {self.qr_code_images} ({percent(self.qr_code_images)})")
        logging.info(f"- 内容区域不足: {self.content_area_rejected} ({percent(self.content_area_rejected)})")
        
        if self.total_images > 0:
            pass_rate = self.passed_images / self.total_images * 100
            logging.info(f"\n图片通过率: {pass_rate:.2f}%")

def reset_processing_state(db):
    """重置所有映射记录的处理状态为未处理，并清空过滤集合"""
    logging.info("正在重置所有映射记录的处理状态...")
    
    for year in years_to_process:
        # 重置映射集合
        mapping_collection_name = f"{year}_mapping"
        if mapping_collection_name in db.list_collection_names():
            mapping_collection = db[mapping_collection_name]
            
            # 更新所有记录的处理状态
            result = mapping_collection.update_many(
                {},  # 匹配所有记录
                {"$set": {"processed": False, "basic_filtered": False}}  # 设置为未处理
            )
            
            logging.info(f"已重置 {year} 年的 {result.modified_count} 条映射记录")
        
        # 清空过滤集合
        filtered_collection_name = f"{year}_filtered"
        if filtered_collection_name in db.list_collection_names():
            db[filtered_collection_name].drop()
            logging.info(f"已清空 {year} 年的过滤集合 {filtered_collection_name}")

def main():
    start_time = time.time()
    logging.info("=== 开始运行图片基础筛选程序 ===")
    
    # 连接MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['sina_news_dataset_test']
    
    # 重置处理状态
    reset_processing_state(db)
    
    # 创建图片筛选器
    filter = ImageBasicFilter()
    
    # 处理每一年的图片
    for year in years_to_process:
        logging.info(f"\n开始处理 {year} 年的图片")
        
        # 处理该年的图片
        filter.process_year_images(year, db)
    
    # 打印总统计
    filter.print_summary()
    
    elapsed_time = time.time() - start_time
    logging.info(f"\n总处理时间: {elapsed_time:.2f} 秒")
    logging.info("=== 基础筛选完成 ===")

if __name__ == "__main__":
    main()
