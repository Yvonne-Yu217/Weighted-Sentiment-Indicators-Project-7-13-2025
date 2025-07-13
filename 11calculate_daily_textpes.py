#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每日加权TextPes指标脚本

该脚本从MongoDB数据库中读取新闻文本内容，分析其情感倾向，
根据公式 WeightedTextPes_t = Σ(Textneg_it × W_i) / Σ(W_i) 计算每日加权TextPes指标。
其中:
- Textneg_it 是第i篇新闻在t日的负面情绪概率
- W_i 是该新闻的质量分数(quality_score)，如果不存在则使用1.0
"""

import os
import sys
import logging
import os
import tempfile

# Workaround for a potential torch._dynamo issue where os.getcwd() fails.
# Set a specific directory for torch compile debug logs.
try:
    torch_debug_dir = os.path.join(tempfile.gettempdir(), "torch_compile_debug")
    os.makedirs(torch_debug_dir, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = torch_debug_dir
except Exception as e:
    # If this fails, it's not critical, but we should log it.
    print(f"Could not set TORCH_COMPILE_DEBUG_DIR: {e}")

import argparse
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# 配置日志
# Set up logging to a file in the script's directory to avoid getcwd issues
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, 'calculate_daily_textpes.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]
)

# 全局配置
BATCH_SIZE = 8  # 批处理大小，适应M1内存
MAX_LENGTH = 512  # 最大文本长度
# 检测是否为M1 Mac并设置设备
is_mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE = torch.device("mps" if is_mps_available else "cpu")

# 使用Erlangshen-Roberta-110M-Sentiment模型
MODEL_NAME = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'

# 情感标签映射（二分类）
SENTIMENT_LABELS = ['负面', '正面']

class NewsDataset(Dataset):
    """新闻文本数据集"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def connect_to_mongodb():
    """连接到MongoDB数据库"""
    try:
        # 连接到MongoDB
        client = MongoClient('localhost', 27017)
        db = client['sina_news_dataset_test']
        logging.info("成功连接到MongoDB")
        return db
    except Exception as e:
        logging.error(f"连接MongoDB失败: {e}")
        sys.exit(1)

def load_model_and_tokenizer():
    """加载预训练模型和分词器"""
    logging.info(f"加载模型: {MODEL_NAME}，使用设备: {DEVICE}")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        model = model.to(DEVICE)
        model.eval()
        logging.info(f"成功加载模型到设备: {DEVICE}")
        return model, tokenizer, 2  # 固定为二分类
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise

def predict_sentiment(model, tokenizer, texts, num_labels):
    """预测文本情感"""
    if not texts:
        return np.array([])
        
    dataset = NewsDataset(texts, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测情感", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            batch_predictions = probs.detach().cpu().numpy()
            predictions.extend(batch_predictions)
    
    return np.array(predictions)

def process_sentiment_with_bert(db, years, source_suffix='_sentiment', target_suffix='_text_sentiment', batch_size=100):
    """
    每次运行时重新分析文章情感，并将结果存储到MongoDB。
    
    参数:
        db: MongoDB数据库连接
        years: 要处理的年份列表
        source_suffix: 源数据集合后缀，默认为'_sentiment'
        target_suffix: 目标数据集合后缀，默认为'_text_sentiment'
        batch_size: 批处理大小
    
    返回:
        处理的文章总数
    """
    # 加载模型和分词器
    model, tokenizer, num_labels = load_model_and_tokenizer()
    total_processed = 0
    
    for year in tqdm(years, desc="总体进度"):
        # 检查源集合是否存在
        source_collection_name = f"{year}{source_suffix}"
        if source_collection_name not in db.list_collection_names():
            logging.warning(f"源集合 '{source_collection_name}' 不存在，跳过年份 {year}")
            continue
            
        # 获取目标集合并重置
        target_collection_name = f"{year}{target_suffix}"
        target_collection = db[target_collection_name]
        logging.info(f"正在重置目标集合: '{target_collection_name}'...")
        target_collection.delete_many({})
        logging.info(f"目标集合 '{target_collection_name}' 已清空。")

        source_collection = db[source_collection_name]
        
        # 获取需要处理的文章总数以用于进度条
        query = {"content": {"$exists": True, "$ne": ""}}
        total_docs_in_year = source_collection.count_documents(query)
        
        if total_docs_in_year == 0:
            logging.info(f"年份 {year} 没有需要处理的文章。")
            continue

        # 从源集合获取所有文章
        cursor = source_collection.find(
            query,
            {"_id": 1, "content": 1, "news_date": 1, "title": 1, "url": 1, "avg_quality_score": 1}
        )
        
        # 分批处理文章
        batch_docs = []
        batch_texts = []
        
        with tqdm(total=total_docs_in_year, desc=f"分析年份 {year}", unit="篇", leave=False) as pbar:
            for doc in cursor:
                batch_docs.append(doc)
                batch_texts.append(doc.get("content", ""))
                
                if len(batch_texts) >= batch_size:
                    predictions = predict_sentiment(model, tokenizer, batch_texts, num_labels)
                    sentiment_docs = []
                    for i, pred in enumerate(predictions):
                        doc_item = batch_docs[i]
                        neg_likelihood = float(pred[0])
                        sentiment_class = 1 if neg_likelihood > 0.5 else 0
                        sentiment_label = "负面" if sentiment_class == 1 else "正面"
                        sentiment_docs.append({
                            "source_id": doc_item["_id"],
                            "news_date": doc_item.get("news_date"),
                            "title": doc_item.get("title", ""),
                            "url": doc_item.get("url", ""),
                            "quality_score": doc_item.get("avg_quality_score", 1.0),
                            "sentiment_class": sentiment_class,
                            "sentiment_label": sentiment_label,
                            "negative_likelihood": neg_likelihood,
                            "processed_at": datetime.now()
                        })
                    
                    if sentiment_docs:
                        target_collection.insert_many(sentiment_docs)
                    
                    pbar.update(len(batch_texts))
                    total_processed += len(batch_texts)
                    batch_docs, batch_texts = [], []

            # 处理剩余的批次
            if batch_texts:
                predictions = predict_sentiment(model, tokenizer, batch_texts, num_labels)
                sentiment_docs = []
                for i, pred in enumerate(predictions):
                    doc_item = batch_docs[i]
                    neg_likelihood = float(pred[0])
                    sentiment_class = 1 if neg_likelihood > 0.5 else 0
                    sentiment_label = "负面" if sentiment_class == 1 else "正面"
                    sentiment_docs.append({
                        "source_id": doc_item["_id"],
                        "news_date": doc_item.get("news_date"),
                        "title": doc_item.get("title", ""),
                        "url": doc_item.get("url", ""),
                        "quality_score": doc_item.get("quality_score", 1.0),
                        "sentiment_class": sentiment_class,
                        "sentiment_label": sentiment_label,
                        "negative_likelihood": neg_likelihood,
                        "processed_at": datetime.now()
                    })
                
                if sentiment_docs:
                    target_collection.insert_many(sentiment_docs)
                
                pbar.update(len(batch_texts))
                total_processed += len(batch_texts)

    logging.info(f"所有年份处理完成: 总共处理 {total_processed} 篇文章")
    return total_processed


def fetch_news_by_date(collection, news_date):
    """从数据库获取指定日期的新闻"""
    # 获取该日期的所有新闻记录
    cursor = collection.find({'news_date': news_date})
    news_list = list(cursor)
    texts = []
    news_ids = []
    
    for news in news_list:
        # 确保有内容字段
        if 'content' in news and news['content']:
            content = news['content'][:MAX_LENGTH * 4]
            texts.append(content)
            news_ids.append(news['_id'])
    
    return texts, news_ids, news_list

def calculate_daily_textpes(db, years, output_file=None, text_collection_suffix='_text_sentiment'):
    """
    计算每日的四种TextPes指标

    指标:
    - TextPes (Binary): 当日负面新闻数 / 总新闻数
    - TextPes_likelihood (Likelihood): 当日新闻负面概率均值
    - WeightedTextPes (Weighted Binary): 按质量分数加权的负面新闻比例
    - W.TextPes_L (Weighted Likelihood): 按质量分数加权的平均负面概率
    
    参数:
        db: MongoDB数据库连接
        years: 要处理的年份列表
        output_file: 输出文件路径
        text_collection_suffix: 文本情感集合后缀，默认为'_text_sentiment'
    
    返回:
        包含每日TextPes指标的DataFrame
    """
    all_results = []
    logging.info(f"开始计算每日TextPes指标，处理年份: {years}")

    for year in tqdm(years, desc="处理年份"):
        collection_name = f"{year}{text_collection_suffix}"
        if collection_name not in db.list_collection_names():
            logging.warning(f"集合 '{collection_name}' 不存在，跳过年份 {year}。")
            continue
        
        collection = db[collection_name]
        logging.info(f"处理集合 '{collection_name}'")

        # MongoDB聚合管道 - 注意使用下划线替代点号以避免MongoDB字段名称限制
        pipeline = [
            {
                "$match": {
                    "sentiment_class": {"$in": [0, 1]},
                    "negative_likelihood": {"$exists": True, "$ne": None},
                    "quality_score": {"$exists": True, "$ne": None},
                    "news_date": {"$exists": True, "$ne": None}
                }
            },
            {
                "$group": {
                    "_id": "$news_date",
                    "total_articles": {"$sum": 1},
                    "negative_articles": {"$sum": "$sentiment_class"},
                    "total_neg_likelihood": {"$sum": "$negative_likelihood"},
                    "total_weight": {"$sum": "$quality_score"},
                    "total_weighted_neg_binary": {"$sum": {"$multiply": ["$sentiment_class", "$quality_score"]}},
                    "total_weighted_neg_likelihood": {"$sum": {"$multiply": ["$negative_likelihood", "$quality_score"]}}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "news_date": "$_id",
                    "TextPes": {"$divide": ["$negative_articles", "$total_articles"]},
                    "TextPes_L": {"$divide": ["$total_neg_likelihood", "$total_articles"]},
                    "W_TextPes": {
                        "$cond": [{"$eq": ["$total_weight", 0]}, 0, {"$divide": ["$total_weighted_neg_binary", "$total_weight"]}]
                    },
                    "W_TextPes_L": {
                        "$cond": [{"$eq": ["$total_weight", 0]}, 0, {"$divide": ["$total_weighted_neg_likelihood", "$total_weight"]}]
                    }
                }
            }
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
            logging.info(f"年份 {year} 计算完成，获取到 {len(results)} 条每日记录")
            if results:
                all_results.extend(results)
        except Exception as e:
            logging.error(f"计算年份 {year} 时出错: {e}")

    if not all_results:
        logging.warning("No data found for the specified years.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    if df.empty:
        logging.warning("DataFrame is empty after processing all years.")
        return df
        
    # 打印一下数据框的列名，便于调试
    logging.info(f"DataFrame columns: {df.columns.tolist()}")
    
    df['news_date'] = pd.to_datetime(df['news_date'])
    df = df.sort_values(by="news_date").reset_index(drop=True)

    # 重命名列以匹配之前的格式
    rename_dict = {}
    if 'TextPes_L' in df.columns:
        rename_dict['TextPes_L'] = 'TextPes_likelihood'
    if 'W_TextPes' in df.columns:
        rename_dict['W_TextPes'] = 'WeightedTextPes'
    if 'W_TextPes_L' in df.columns:
        rename_dict['W_TextPes_L'] = 'W.TextPes_L'
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logging.info(f"列重命名完成: {rename_dict}")
    
    # 检查所有必需的列是否存在
    final_columns = ['news_date', 'TextPes', 'TextPes_likelihood', 'WeightedTextPes', 'W.TextPes_L']
    missing_columns = [col for col in final_columns if col not in df.columns]
    
    if missing_columns:
        logging.warning(f"Missing columns in DataFrame: {missing_columns}")
        # 为缺失的列添加默认值(NaN)
        for col in missing_columns:
            df[col] = float('nan')
    df = df[final_columns]

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved daily TextPes indicators to {output_file}")
        
    return df

def update_excel_with_textpes(df, excel_file):
    """
    更新Excel文件，添加所有TextPes指标
    
    参数:
        df: 包含TextPes指标的DataFrame
        excel_file: Excel文件路径
    """
    if df is None or df.empty:
        logging.warning("没有数据可供更新Excel")
        return
    
    try:
        # 检查Excel文件是否存在
        if not os.path.exists(excel_file):
            logging.error(f"Excel文件不存在: {excel_file}")
            return
        
        # 读取Excel文件
        logging.info(f"正在读取Excel文件: {excel_file}")
        df_excel = pd.read_excel(excel_file)
        
        # 确保Excel中有Date列
        if "Date" not in df_excel.columns:
            logging.error("Excel文件中没有Date列")
            return
        
        # 确保Date列是日期类型
        df_excel["Date"] = pd.to_datetime(df_excel["Date"])
        
        # 重命名df中的news_date列为Date，以便合并
        df_merge = df.copy()
        df_merge = df_merge.rename(columns={"news_date": "Date"})
        
        # 选择要合并的所有四种指标列
        merge_columns = ['Date', 'TextPes', 'TextPes_likelihood', 'WeightedTextPes', 'W.TextPes_L']
        
        # 检查列是否存在，只选择存在的列
        missing_cols = [col for col in merge_columns if col not in df_merge.columns]
        if missing_cols:
            logging.warning(f"合并Excel时缺少以下列: {missing_cols}")
            available_columns = [col for col in merge_columns if col in df_merge.columns]
            df_merge = df_merge[available_columns]
        else:
            df_merge = df_merge[merge_columns]
        
        # 合并数据，如果列已存在则先删除
        existing_cols_to_drop = [col for col in merge_columns if col in df_excel.columns and col != 'Date']
        if existing_cols_to_drop:
            df_excel = df_excel.drop(columns=existing_cols_to_drop)

        df_updated = pd.merge(df_excel, df_merge, on="Date", how="left")
        
        # 保存更新后的Excel文件
        output_file = excel_file.replace(".xlsx", "_with_TextPes.xlsx")
        df_updated.to_excel(output_file, index=False, engine='openpyxl')
        
        logging.info(f"已将所有TextPes指标添加到Excel文件: {output_file}")
        
        # 打印更新统计
        total_rows = len(df_excel)
        updated_rows = df_updated["WeightedTextPes"].notna().sum()
        logging.info(f"总行数: {total_rows}")
        logging.info(f"更新行数: {updated_rows}")
        logging.info(f"更新比例: {updated_rows/total_rows*100:.2f}%")
        
    except Exception as e:
        logging.error(f"更新Excel文件时出错: {e}")

def fix_sentiment_class_label_inversion(db, years, collection_suffix='_text_sentiment'):
    """
    修正 sentiment_label 与 sentiment_class 的对应关系：
    - sentiment_label 为“正面”的 sentiment_class 改为 0
    - sentiment_label 为“负面”的 sentiment_class 改为 1
    """
    for year in years:
        collection_name = f"{year}{collection_suffix}"
        if collection_name not in db.list_collection_names():
            continue
        collection = db[collection_name]
        # 正面 -> 0
        result1 = collection.update_many(
            {"sentiment_label": "正面", "sentiment_class": {"$ne": 0}},
            {"$set": {"sentiment_class": 0}}
        )
        # 负面 -> 1
        result2 = collection.update_many(
            {"sentiment_label": "负面", "sentiment_class": {"$ne": 1}},
            {"$set": {"sentiment_class": 1}}
        )
        logging.info(f"{collection_name}: 修正正面为0 {result1.modified_count} 条，负面为1 {result2.modified_count} 条")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算每日加权TextPes指标")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2015)),
                        help="要处理的年份列表")
    parser.add_argument("--output", type=str, default="data/weighted_textpes.csv",
                        help="输出文件路径")
    parser.add_argument("--update-excel", type=str, default=None,
                        help="要更新的Excel文件路径")
    parser.add_argument("--source-suffix", type=str, default="_sentiment",
                        help="源数据集合后缀，默认为'_sentiment'")
    parser.add_argument("--target-suffix", type=str, default="_text_sentiment",
                        help="目标情感集合后缀，默认为'_text_sentiment'")
    parser.add_argument("--skip-sentiment-analysis", action="store_true",
                        help="跳过情感分析步骤，直接使用已存在的情感分析结果")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="批处理大小，默认为100")
    
    args = parser.parse_args()
    
    # 连接数据库
    db = connect_to_mongodb()
    
    # 先使用BERT模型分析文章情感并存储结果
    if not args.skip_sentiment_analysis:
        logging.info("开始使用BERT模型分析文章情感...")
        processed_count = process_sentiment_with_bert(
            db, args.years, 
            source_suffix=args.source_suffix, 
            target_suffix=args.target_suffix,
            batch_size=args.batch_size
        )
        logging.info(f"情感分析完成，共处理 {processed_count} 篇文章")
    else:
        logging.info("跳过情感分析步骤，直接计算指标")
    
    # 修正 sentiment_class 和 sentiment_label 的对应关系
    fix_sentiment_class_label_inversion(db, args.years, args.target_suffix)
    
    # 准备输出路径，确保使用绝对路径以避免 getcwd 问题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)

    # 计算每日加权TextPes指标
    logging.info("开始计算每日加权TextPes指标...")
    df = calculate_daily_textpes(db, args.years, output_path, args.target_suffix)
    
    # 更新Excel文件
    if args.update_excel and df is not None and not df.empty:
        logging.info(f"更新Excel文件: {args.update_excel}")
        update_excel_with_textpes(df, args.update_excel)
    
    logging.info("所有处理完成")

if __name__ == "__main__":
    main()
