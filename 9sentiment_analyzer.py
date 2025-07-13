import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pymongo import MongoClient
import os
import logging
from pathlib import Path
import datetime
import re
from tqdm import tqdm
import timm
import sys
import glob
import numpy as np

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

# 删除所有._开头的隐藏文件
def remove_all_hidden_files():
    """删除所有._开头的隐藏文件"""
    count = 0
    for root, dirs, files in os.walk('images'):
        for file in files:
            if file.startswith('._'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logging.error(f"删除隐藏文件失败 {file_path}: {e}")
    
    if count > 0:
        logging.info(f"已删除 {count} 个._开头的隐藏文件")

# 改进的ViT模型
class ImprovedViTModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super().__init__()
        # 使用预训练的ViT模型
        self.backbone = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        
        # 获取特征维度
        in_features = self.backbone.head.in_features
        
        # 替换分类头为更复杂的结构
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def check_environment():
    """检查运行环境"""
    logging.info("\n=== 环境检查 ===")
    logging.info(f"PyTorch 版本: {torch.__version__}")
    logging.info(f"CUDA 可用: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        logging.info(f"MPS 可用: {torch.backends.mps.is_available()}")
    
    # 检查必要目录
    required_dirs = ['images']
    for d in required_dirs:
        if os.path.exists(d):
            logging.info(f"目录存在: {d}")
        else:
            logging.info(f"目录不存在: {d}")
    logging.info("=== 检查完成 ===\n")

def load_model(model_path):
    """加载训练好的模型"""
    try:
        # 选择合适的设备
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("使用 CUDA 设备")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("使用 MPS 设备")
        else:
            device = torch.device("cpu")
            logging.info("使用 CPU 设备")
        
        # 创建模型实例
        model = ImprovedViTModel()
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        logging.info(f"模型 {model_path} 加载成功")
        return model, device
        
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        raise

def classify_image(model, image_path, device):
    """对单张图片进行分类"""
    try:
        # 图像预处理 - 与训练时完全相同
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载并转换图像
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
            _, predicted = torch.max(outputs, 1)
            
            # 返回预测类别和概率
            # 注意：positive=0, negative=1
            return predicted.item(), probs[1]  # 返回类别和负面概率
            
    except Exception as e:
        logging.error(f"处理图片失败 {image_path}: {e}")
        return None, None

def connect_to_mongodb():
    """连接到 MongoDB 数据库"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['sina_news_dataset_test']  # 使用项目数据库名
        
        # 检查数据库连接
        client.server_info()
        logging.info("MongoDB 连接成功")
        return db
        
    except Exception as e:
        logging.error(f"MongoDB 连接失败: {e}")
        raise

def remove_hidden_files(directory):
    """删除._开头的隐藏文件"""
    directory = Path(directory)
    count = 0
    for file in directory.glob("._*"):
        file.unlink()
        count += 1
    if count > 0:
        logging.info(f"已删除 {count} 个._开头的隐藏文件")

def check_sentiment_collection(db, year):
    """检查情感分析集合的状态，并提供详细的诊断信息"""
    logging.info(f"对 {year} 年的 sentiment 集合进行双重检查...")
    collection_name = f"{year}_sentiment"
    collection = db.get_collection(collection_name)

    if collection is None:
        logging.error(f"集合 {collection_name} 不存在，请先运行图片处理流程。")
        return False

    query = {'all_images.use_for_sentiment': True}

    # 检查1: 使用 count_documents
    try:
        count = collection.count_documents(query)
        logging.info(f"[检查1 - count_documents] 查询 {{'all_images.use_for_sentiment': True}}，找到 {count} 条记录。")
    except Exception as e:
        logging.error(f"[检查1 - count_documents] 查询时出错: {e}")
        return False

    # 检查2: 使用 find_one
    try:
        doc = collection.find_one(query)
        if doc:
            logging.info(f"[检查2 - find_one] 成功找到一条匹配记录 (ID: {doc.get('_id')})。")
        else:
            logging.warning("[检查2 - find_one] 未找到任何匹配记录。")
    except Exception as e:
        logging.error(f"[检查2 - find_one] 查询时出错: {e}")
        return False

    # 最终判断
    if count > 0 or doc is not None:
        logging.info(f"检查通过，集合 {collection_name} 中存在待处理的图片。")
        return True
    else:
        logging.warning(f"检查未通过，集合 {collection_name} 中确实没有标记为 use_for_sentiment=True 的图片。")
        return False

def calculate_enhanced_weight(clarity_weight, text_weight):
    """
    计算增强版的图片权重，使用加权平均和非线性变换
    
    参数:
    clarity_weight: 清晰度权重
    text_weight: 文字权重
    
    返回:
    增强后的权重值
    """
    # 1. 基础权重计算 - 加权平均
    text_importance = 0.6  # 文字因素更重要
    base_weight = clarity_weight * (1-text_importance) + text_weight * text_importance
    
    # 2. 非线性变换增强对比度
    enhanced_weight = np.tanh(base_weight * 1.2)
    
    return float(enhanced_weight)

def process_images_for_year(model, device, db, year):
    """处理指定年份的图片"""
    logging.info(f"\n开始处理 {year} 年的图片...")

    # 检查集合是否存在，并获取文档总数
    if not check_sentiment_collection(db, year):
        return

    sentiment_collection = db[f"{year}_sentiment"]
    mapping_collection_name = f"{year}_mapping"
    mapping_collection = db.get_collection(mapping_collection_name)
    if mapping_collection is None:
        logging.warning(f"'{mapping_collection_name}' 集合不存在，部分图片路径可能无法解析")

    # 根据用户要求，在分析前重置情感分析字段
    logging.info(f"Resetting sentiment fields for year {year} in collection '{sentiment_collection.name}'...")
    update_result = sentiment_collection.update_many(
        {"all_images": {"$exists": True}},
        {
            "$unset": {
                "all_images.$[].sentiment_score": "",
                "all_images.$[].predicted_class": "",
                "all_images.$[].negative_likelihood": "",
                "all_images.$[].positive_likelihood": "",
                "all_images.$[].image_weight": "",
            }
        }
    )
    logging.info(f"Reset complete. Matched {update_result.matched_count} documents, modified {update_result.modified_count} documents.")

    # 初始化缓存和图像预处理器
    folder_path_cache = {}
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 初始化计数器
    stats = {
        "processed": 0, "failed": 0, "skipped": 0,
        "negative": 0, "positive": 0,
        "total_negative_prob": 0.0, "weighted_negative_prob": 0.0, "total_weight": 0.0
    }
    failed_image_details = []

    # 遍历所有新闻文档
    cursor = sentiment_collection.find({})
    total_docs = sentiment_collection.count_documents({})
    
    if total_docs == 0:
        logging.warning(f"{year} 年的 sentiment 集合中没有文档。")
        return

    with tqdm(total=total_docs, desc=f"处理 {year} 年新闻") as pbar:
        for doc in cursor:
            original_id = doc.get("original_id")
            all_images = doc.get("all_images", [])

            if not all_images or not original_id:
                pbar.update(1)
                continue

            # 获取并缓存 folder_path
            img_folder_path = folder_path_cache.get(original_id)
            if not img_folder_path and mapping_collection is not None:
                try:
                    # 使用 original_id 进行查询
                    mapping_doc = mapping_collection.find_one({"original_id": original_id})
                    if mapping_doc and "folder_path" in mapping_doc:
                        img_folder_path = mapping_doc["folder_path"]
                        folder_path_cache[original_id] = img_folder_path
                except Exception as e:
                    logging.warning(f"数据库查询失败 (original_id: {original_id}): {e}")

            # 遍历文档中的每张图片
            for img_index, img_obj in enumerate(all_images):
                # 条件1: 必须标记为 use_for_sentiment
                # 条件2: 不能是已经分析过的
                if not img_obj.get("use_for_sentiment") or "predicted_class" in img_obj:
                    stats["skipped"] += 1
                    continue

                # --- 核心路径解析逻辑 ---
                image_file_path = None
                # 优先级1: actual_path
                if img_obj.get("actual_path") and os.path.exists(img_obj["actual_path"]):
                    image_file_path = img_obj["actual_path"]
                # 优先级2: folder_path + path
                elif img_folder_path and img_obj.get("path"):
                    potential_path = os.path.join(img_folder_path, img_obj["path"])
                    if os.path.exists(potential_path):
                        image_file_path = potential_path
                
                if not image_file_path:
                    stats["failed"] += 1
                    failed_image_details.append((img_obj.get("path", f"doc_{doc['_id']}_img_{img_index}"), "无法解析有效路径"))
                    continue

                # --- 图片分类与数据库更新 ---
                try:
                    image = Image.open(image_file_path).convert('RGB')
                    image_tensor = preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs[0], dim=0)
                        # 确保正确获取消极概率（1表示消极，0表示积极）
                        negative_prob = probabilities[1].item()
                        # 0 表示积极，1 表示消极
                    predicted_class = 1 if negative_prob > 0.5 else 0

                    # 更新统计数据
                    # 更新计数（predicted_class: 0=积极，1=消极）
                    stats["positive" if predicted_class == 0 else "negative"] += 1
                    stats["total_negative_prob"] += negative_prob
                    weight = calculate_enhanced_weight(img_obj.get("clarity_weight", 1.0), img_obj.get("text_weight", 1.0))
                    stats["total_weight"] += weight
                    stats["weighted_negative_prob"] += negative_prob * weight
                    
                    # 精确更新数据库中的嵌套对象
                    update_field_prefix = f"all_images.{img_index}"
                    sentiment_collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {
                            f"{update_field_prefix}.predicted_class": predicted_class,
                            f"{update_field_prefix}.negative_likelihood": negative_prob,
                            f"{update_field_prefix}.image_file_path": image_file_path # 保存最终使用的路径
                        }}
                    )
                    stats["processed"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    failed_image_details.append((image_file_path, str(e)))
            
            pbar.update(1)

    # --- 最终统计与聚合 ---
    logging.info("\n=== 处理统计 ===")
    logging.info(f"已处理新闻文档数: {total_docs}")
    logging.info(f"成功处理图片数: {stats['processed']}")
    logging.info(f"失败图片数: {stats['failed']}")
    logging.info(f"跳过图片数 (不需处理或已处理): {stats['skipped']}")
    logging.info(f"folder_path 缓存大小: {len(folder_path_cache)} 条记录")

    if stats["processed"] > 0:
        logging.info(f"负面图片数: {stats['negative']} ({(stats['negative']/stats['processed']):.2%})")
        logging.info(f"正面图片数: {stats['positive']} ({(stats['positive']/stats['processed']):.2%})")
        logging.info(f"平均负面概率: {(stats['total_negative_prob']/stats['processed']):.4f}")
        if stats["total_weight"] > 0:
            logging.info(f"加权平均负面概率: {(stats['weighted_negative_prob']/stats['total_weight']):.4f}")

    if failed_image_details:
        logging.warning("\n--- 部分失败记录 (最多显示10条) ---")
        for path, error in failed_image_details[:10]:
            logging.warning(f"- 路径: {path}, 原因: {error}")

    # 计算新闻级别的平均情感分数
    if stats["processed"] > 0:
        logging.info("\n正在计算新闻级别的平均情感分数...")
        news_sentiment_collection = db[f"{year}_news_sentiment"]
        news_sentiment_collection.drop()
        pipeline = [
            {"$unwind": "$all_images"},
            {"$match": {"all_images.predicted_class": {"$exists": True}}},
            {"$group": {
                "_id": {"news_date": "$news_date", "title": "$title", "original_id": "$original_id"},
                "avg_negative_likelihood": {"$avg": "$all_images.negative_likelihood"},
                "image_count": {"$sum": 1},
                "negative_count": {"$sum": {"$cond": [{"$eq": ["$all_images.predicted_class", 1]}, 1, 0]}},
                "positive_count": {"$sum": {"$cond": [{"$eq": ["$all_images.predicted_class", 0]}, 1, 0]}}
            }},
            {"$project": {
                "_id": 0,
                "original_id": "$_id.original_id",
                "news_date": "$_id.news_date",
                "title": "$_id.title",
                "avg_negative_likelihood": 1,
                "image_count": 1,
                "negative_count": 1,
                "positive_count": 1,
                "negative_ratio": {"$divide": ["$negative_count", "$image_count"]}
            }},
            {"$sort": {"news_date": 1}}
        ]
        
        try:
            news_results = list(sentiment_collection.aggregate(pipeline))
            if news_results:
                news_sentiment_collection.insert_many(news_results)
                logging.info(f"已计算并保存 {len(news_results)} 条新闻的平均情感分数。")
        except Exception as e:
            logging.error(f"聚合新闻级别情感分数时出错: {e}")
def main():
    """主函数，提供预测功能"""
    try:
        import argparse
        
        # 过滤掉Jupyter相关的参数
        jupyter_args = [arg for arg in sys.argv if arg.startswith('--f=')]
        for arg in jupyter_args:
            sys.argv.remove(arg)
        
        parser = argparse.ArgumentParser(description='ViT情感分析模型预测')
        parser.add_argument('--years', type=int, nargs='+', default=years_to_process,
                            help='要处理的年份列表，例如: --years 2019 2022 2024')
        parser.add_argument('--model', type=str, default='improved_vit_sentiment_model.pth',
                            help='模型路径，默认使用improved_vit_sentiment_model.pth')
        
        args = parser.parse_args()
        
        # 检查环境
        check_environment()
        
        # 删除隐藏文件
        # remove_all_hidden_files() # 暂时禁用，此函数在大量文件时可能非常缓慢
        
        # 获取当前工作目录
        current_dir = os.getcwd()
        logging.info(f"当前工作目录: {current_dir}")
        
        # 检查模型文件
        if not os.path.exists(args.model):
            logging.error(f"模型文件不存在: {args.model}")
            return
        
        # 加载模型
        model, device = load_model(args.model)
        
        # 连接数据库
        db = connect_to_mongodb()
        
        # 处理每年的图片
        for year in args.years:
            process_images_for_year(model, device, db, year)
        
        logging.info("\n情感分析完成，结果已保存到 MongoDB")
        
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
        raise

if __name__ == "__main__":
    main()
