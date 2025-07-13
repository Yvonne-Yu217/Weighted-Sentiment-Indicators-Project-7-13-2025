import os
import cv2
import numpy as np
import logging
import time
import pytesseract
from PIL import Image
from tqdm import tqdm
from pymongo import MongoClient
from typing import Dict, Any, List

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
BLUR_THRESHOLD = 100  # 调整模糊检测阈值，稍微宽松
MIN_TEXT_LENGTH = 50  # 降低文本检测阈值，文字较少的图片也可能有价值

# 文字量级别阈值（仅用于统计和分类）
TEXT_LEVEL_THRESHOLDS = {
    "none": 0,      # 无文字
    "low": 50,      # 少量文字
    "medium": 200,  # 中等文字
    "high": 290     # 大量文字（基于95%分位数）
}

# 图片质量评分阈值
QUALITY_SCORE_THRESHOLD = 0.5  # 图片质量评分阈值

# 导入权重计算函数
try:
    from weight_function_code import calculate_sentiment_weight
except ImportError:
    # 如果导入失败，提供一个默认的权重计算函数
    def calculate_sentiment_weight(text_length):
        """根据文字长度计算情感权重（默认函数）"""
        midpoint = 290.0  # 95%分位数
        steepness = 0.01  # 陡峭程度
        min_weight = 0.1  # 最小权重
        max_weight = 1.0  # 最大权重
        # 反向权重：文字越少，权重越高
        weight = max_weight - (max_weight - min_weight) / (1 + np.exp(-steepness * (text_length - midpoint)))
        return float(weight)  # 确保返回Python原生float类型

# 导入清晰度权重计算函数
try:
    from clarity_weight_function import calculate_clarity_weight
except ImportError:
    # 如果导入失败，提供一个默认的清晰度权重计算函数
    def calculate_clarity_weight(blur_score):
        """根据图片清晰度分数计算权重（默认函数）"""
        midpoint = 150.0  # 清晰度中点值
        steepness = 0.015  # 曲线陡峭程度
        min_weight = 0.1  # 最小权重
        max_weight = 1.0  # 最大权重
        # 使用逻辑函数计算权重 - 越清晰(blur_score越高)权重越高
        weight = min_weight + (max_weight - min_weight) / (1 + np.exp(-steepness * (blur_score - midpoint)))
        return float(weight)  # 确保返回Python原生float类型

class ImageQualityProcessor:
    """图片质量处理器：处理通过基础筛选的图片，进行质量评分"""
    
    def __init__(self):
        """初始化处理器"""
        # 统计信息
        self.total_images = 0
        self.high_quality_images = 0
        self.low_quality_images = 0
        
        # 权重统计
        self.clarity_weights = []
        self.text_weights = []
        self.quality_scores = []
        
        # 文字量统计
        self.text_level_counts = {
            "none": 0,
            "low": 0,
            "medium": 0,
            "high": 0
        }
    
    def get_blur_score(self, image_path: str) -> float:
        """计算图片清晰度分数"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return 0
                
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 计算拉普拉斯变换
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 返回清晰度分数
            return laplacian_var
            
        except Exception as e:
            logging.error(f"计算清晰度分数出错 {image_path}: {e}")
            return 0
    
    def analyze_text(self, image_path: str) -> Dict[str, Any]:
        """从图片中提取文字并返回统计信息"""
        try:
            # 读取图片
            img = Image.open(image_path)
            # OCR识别
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            cleaned_text = ' '.join(text.split())
            text_length = len(cleaned_text)
            img.close()

            # 文字量级别
            text_level = "none"
            if text_length >= TEXT_LEVEL_THRESHOLDS["high"]:
                text_level = "high"
            elif text_length >= TEXT_LEVEL_THRESHOLDS["medium"]:
                text_level = "medium"
            elif text_length >= TEXT_LEVEL_THRESHOLDS["low"]:
                text_level = "low"
        
            # 计算文字权重
            weight = calculate_sentiment_weight(text_length)
            is_text_image = bool(text_length > MIN_TEXT_LENGTH)  # 转换为Python原生布尔类型
        
            return {
                "text": cleaned_text,
                "text_length": text_length,
                "text_level": text_level,
                "weight": weight,
                "is_text_image": is_text_image
            }
        except Exception as e:
            logging.error(f"文字分析出错 {image_path}: {e}")
            return {
                "text": "",
                "text_length": 0,
                "text_level": "none",
                "weight": 1.0,  # 无文字时最高权重
                "is_text_image": False
            }
    
    def calculate_quality_score(self, blur_score: float, text_info: Dict[str, Any]) -> float:
        """计算图片综合质量分数"""
        # 计算清晰度权重
        clarity_weight = calculate_clarity_weight(blur_score)
        
        # 获取文字权重
        text_weight = text_info["weight"]
        
        # 计算综合质量分数（清晰度权重 * 文字权重）
        quality_score = clarity_weight * text_weight
        
        return quality_score
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """处理单个图片，计算质量分数"""
        self.total_images += 1
        
        try:
            # 计算模糊度分数
            blur_score = self.get_blur_score(image_path)
            
            # 分析图片中的文字
            text_info = self.analyze_text(image_path)
            self.text_level_counts[text_info["text_level"]] += 1
            
            # 计算清晰度权重
            clarity_weight = calculate_clarity_weight(blur_score)
            self.clarity_weights.append(clarity_weight)
            
            # 获取文字权重
            text_weight = text_info["weight"]
            self.text_weights.append(text_weight)
            
            # 计算综合质量分数
            quality_score = self.calculate_quality_score(blur_score, text_info)
            self.quality_scores.append(quality_score)
            
            # 判断是否为高质量图片
            is_high_quality = bool(quality_score >= QUALITY_SCORE_THRESHOLD)
            
            if is_high_quality:
                self.high_quality_images += 1
            else:
                self.low_quality_images += 1
            
            # 返回处理结果，确保所有值都是Python原生类型
            return {
                "valid": True,
                "blur_score": float(blur_score),
                "clarity_weight": float(clarity_weight),
                "text_weight": float(text_weight),
                "text_level": text_info["text_level"],
                "text_length": int(text_info["text_length"]),
                "quality_score": float(quality_score),
                "is_high_quality": is_high_quality
            }
            
        except Exception as e:
            logging.error(f"处理图片出错 {image_path}: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def process_year_images(self, year, db):
        """处理指定年份通过基础筛选的图片"""
        # 获取过滤后的集合
        filtered_collection_name = f"{year}_filtered"
        
        # 检查集合是否存在
        if filtered_collection_name not in db.list_collection_names():
            logging.warning(f"过滤集合 {filtered_collection_name} 不存在，跳过处理")
            return
        
        filtered_collection = db[filtered_collection_name]
        
        # 创建或获取最终结果集合
        result_collection_name = f"{year}_2"
        result_collection = db[result_collection_name]
        
        # 查询条件
        query = {
            "basic_filtered": True,
            "quality_processed": False,
            "has_valid_images": True
        }
        
        # 查询需要处理的记录总数
        total_docs = filtered_collection.count_documents(query)
        
        if total_docs == 0:
            logging.info(f"{year} 年没有需要进行质量处理的记录")
            return
        
        logging.info(f"开始质量处理 {year} 年的 {total_docs} 条记录")
        
        # 批量处理参数
        batch_size = 100  # 每批处理的记录数量
        processed_count = 0
        
        # 使用进度条跟踪处理进度
        with tqdm(total=total_docs, desc=f"质量处理 {year} 年图片") as pbar:
            # 分批处理记录，避免游标超时
            while processed_count < total_docs:
                # 获取一批记录
                batch_cursor = filtered_collection.find(query).limit(batch_size)
                batch_docs = list(batch_cursor)  # 将游标结果转换为列表，避免游标超时
                
                if not batch_docs:  # 如果没有文档了，结束循环
                    break
                
                # 处理这一批记录
                for doc in batch_docs:
                    try:
                        # 获取文档信息
                        original_id = doc.get("original_id")
                        valid_images = doc.get("valid_images", [])
                        
                        # 初始化图片列表
                        processed_images = []
                        high_quality_images = []
                        
                        # 处理每张有效图片
                        for img_info in valid_images:
                            image_path = img_info.get("abs_path")
                            rel_path = img_info.get("path")
                            
                            if os.path.exists(image_path):
                                # 处理图片
                                result = self.process_single_image(image_path)
                                
                                if result["valid"]:
                                    # 将处理结果添加到列表
                                    img_result = {
                                        "path": rel_path,
                                        "clarity_weight": result["clarity_weight"],
                                        "text_weight": result["text_weight"],
                                        "text_level": result["text_level"],
                                        "text_length": result["text_length"],
                                        "quality_score": result["quality_score"],
                                        "is_high_quality": result["is_high_quality"]
                                    }
                                    processed_images.append(img_result)
                                    
                                    # 如果是高质量图片，添加到高质量列表
                                    if result["is_high_quality"]:
                                        high_quality_images.append(img_result)
                        
                        # 创建新的文档，包含质量处理结果
                        new_doc = {
                            "original_id": original_id,
                            "title": doc.get("title", ""),
                            "content": doc.get("content", ""),
                            "news_date": doc.get("news_date", ""),
                            "category": doc.get("category", ""),
                            "link": doc.get("link", ""),
                            "all_images": processed_images,
                            "high_quality_images": high_quality_images,
                            "has_high_quality_images": len(high_quality_images) > 0,
                            "avg_clarity_weight": sum([img["clarity_weight"] for img in processed_images]) / len(processed_images) if processed_images else 0,
                            "avg_text_weight": sum([img["text_weight"] for img in processed_images]) / len(processed_images) if processed_images else 0,
                            "avg_quality_score": sum([img["quality_score"] for img in processed_images]) / len(processed_images) if processed_images else 0
                        }
                        
                        # 插入或更新到结果集合
                        result_collection.update_one(
                            {"original_id": original_id},
                            {"$set": new_doc},
                            upsert=True
                        )
                        
                        # 更新过滤集合中的记录状态
                        filtered_collection.update_one(
                            {"_id": doc["_id"]},
                            {"$set": {"quality_processed": True}}
                        )
                        
                    except Exception as e:
                        logging.error(f"处理记录出错: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # 更新进度条和计数器
                    pbar.update(1)
                    processed_count += 1
    def print_summary(self):
        logging.info("\n文字量分布:")
        logging.info(f"- 无文字: {self.text_level_counts['none']}")
        logging.info(f"- 少量文字: {self.text_level_counts['low']}")
        logging.info(f"- 中等文字: {self.text_level_counts['medium']}")
        logging.info(f"- 大量文字: {self.text_level_counts['high']}")
        
        if self.clarity_weights:
            logging.info("\n清晰度权重统计:")
            logging.info(f"- 平均值: {np.mean(self.clarity_weights):.4f}")
            logging.info(f"- 中位数: {np.median(self.clarity_weights):.4f}")
            logging.info(f"- 最小值: {min(self.clarity_weights):.4f}")
            logging.info(f"- 最大值: {max(self.clarity_weights):.4f}")
        
        if self.text_weights:
            logging.info("\n文字权重统计:")
            logging.info(f"- 平均值: {np.mean(self.text_weights):.4f}")
            logging.info(f"- 中位数: {np.median(self.text_weights):.4f}")
            logging.info(f"- 最小值: {min(self.text_weights):.4f}")
            logging.info(f"- 最大值: {max(self.text_weights):.4f}")
        
        if self.quality_scores:
            logging.info("\n质量分数统计:")
            logging.info(f"- 平均值: {np.mean(self.quality_scores):.4f}")
            logging.info(f"- 中位数: {np.median(self.quality_scores):.4f}")
            logging.info(f"- 最小值: {min(self.quality_scores):.4f}")
            logging.info(f"- 最大值: {max(self.quality_scores):.4f}")
        
        if self.total_images > 0:
            high_quality_rate = self.high_quality_images / self.total_images * 100
            logging.info(f"\n高质量图片比例: {high_quality_rate:.2f}%")

def reset_quality_processing_state(db):
    """重置质量处理状态，清空结果集合"""
    logging.info("正在重置图片质量处理状态...")
    
    for year in years_to_process:
        # 重置filtered集合中的quality_processed标志
        filtered_collection_name = f"{year}_filtered"
        if filtered_collection_name in db.list_collection_names():
            filtered_collection = db[filtered_collection_name]
            result = filtered_collection.update_many(
                {"quality_processed": True},
                {"$set": {"quality_processed": False}}
            )
            logging.info(f"已重置 {year} 年的 {result.modified_count} 条质量处理记录")
        
        # 清空或重建结果集合
        result_collection_name = f"{year}_2"
        if result_collection_name in db.list_collection_names():
            db[result_collection_name].drop()
            logging.info(f"已清空 {year} 年的结果集合 {result_collection_name}")

def main():
    start_time = time.time()
    logging.info("=== 开始运行图片质量处理程序 ===")
    
    # 连接MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['sina_news_dataset_test']
    
    # 重置质量处理状态
    reset_quality_processing_state(db)
    
    # 创建图片处理器
    processor = ImageQualityProcessor()
    
    # 处理每一年的图片
    for year in years_to_process:
        logging.info(f"\n开始处理 {year} 年的图片")
        
        # 处理该年的图片
        processor.process_year_images(year, db)
    
    # 打印总统计
    processor.print_summary()
    
    elapsed_time = time.time() - start_time
    logging.info(f"\n总处理时间: {elapsed_time:.2f} 秒")
    logging.info("=== 程序运行完成 ===")

if __name__ == "__main__":
    main()
