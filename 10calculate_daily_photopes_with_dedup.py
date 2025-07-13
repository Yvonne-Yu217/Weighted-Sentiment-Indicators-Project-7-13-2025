#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每日加权PhotoPes指标脚本（去重版本）

该脚本从MongoDB数据库中读取图片情感分析结果，
根据公式 WeightedPhotoPes_t = Σ(Neg_it × W_i) / Σ(W_i) 计算每日加权PhotoPes指标。
其中:
- Neg_it 是第i张图片在t日的负面情绪概率
- W_i 是该图片的质量分数(quality_score)

该版本只使用标记为use_for_sentiment=True的图片进行计算，
即非重复图片或每组重复图片中质量最高的图片。
"""

import os
import sys
import logging
import argparse
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calculate_daily_photopes.log', encoding='utf-8')
    ]
)

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

def calculate_daily_photopes(db, years, output_file=None):
    """
    计算每日的四种PhotoPes指标 (去重版本)
    
    指标:
    - PhotoPes (Binary): 当日负面图片数 / 总图片数
    - PhotoPes_L (Likelihood): 当日图片负面概率均值
    - W.PhotoPes (Weighted Binary): 按图片质量加权的负面图片比例
    - W.PhotoPes_L (Weighted Likelihood): 按图片质量加权的平均负面概率
    
    参数:
        db: MongoDB数据库连接
        years: 要处理的年份列表
        output_file: 输出文件路径
    
    返回:
        包含每日PhotoPes指标的DataFrame
    """
    all_results = []
    
    for year in tqdm(years, desc="Processing years"):
        collection_name = f"{year}_sentiment"
        if collection_name not in db.list_collection_names():
            logging.warning(f"Collection '{collection_name}' not found, skipping year {year}.")
            continue
            
        collection = db[collection_name]
        
        # MongoDB聚合管道，用于高效计算所有指标
        # 修正聚合管道
        pipeline = [
            # 首先展开all_images数组
            {"$unwind": "$all_images"},
            # 筛选出用于情感分析的图片记录
            {"$match": {
                "all_images.use_for_sentiment": True,
                "all_images.predicted_class": {"$exists": True},
                "all_images.quality_score": {"$exists": True, "$ne": None},
                "all_images.negative_likelihood": {"$exists": True, "$ne": None},
                "news_date": {"$exists": True, "$ne": None}
            }},
            # 按日期分组并计算指标
            {"$group": {
                "_id": "$news_date",
                "total_images": {"$sum": 1},
                # 正确的标签：消极=1，积极=0。直接求和即得到消极图片数。
                "negative_images": {"$sum": "$all_images.predicted_class"},
                "total_neg_likelihood": {"$sum": "$all_images.negative_likelihood"},
                "total_weight": {"$sum": "$all_images.quality_score"},
                # 正确的标签处理：权重乘以predicted_class，得到消极图片的加权分数
                "total_weighted_neg_binary": {"$sum": {
                    "$multiply": ["$all_images.predicted_class", "$all_images.quality_score"]
                }},
                "total_weighted_neg_likelihood": {"$sum": {
                    "$multiply": ["$all_images.negative_likelihood", "$all_images.quality_score"]
                }}
            }},
            # 计算最终的四个指标（避免使用带点号的字段名）
            {"$project": {
                "_id": 0,
                "news_date": "$_id",
                "PhotoPes": {"$divide": ["$negative_images", "$total_images"]},
                "PhotoPes_L": {"$divide": ["$total_neg_likelihood", "$total_images"]},
                "W_PhotoPes": {
                    "$cond": [{"$eq": ["$total_weight", 0]}, 0, {"$divide": ["$total_weighted_neg_binary", "$total_weight"]}]
                },
                "W_PhotoPes_L": {
                    "$cond": [{"$eq": ["$total_weight", 0]}, 0, {"$divide": ["$total_weighted_neg_likelihood", "$total_weight"]}]
                }
            }}
        ]
        
        try:
            results = list(collection.aggregate(pipeline))
            if results:
                all_results.extend(results)
        except Exception as e:
            logging.error(f"Error during aggregation for year {year}: {e}")

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
    if 'PhotoPes_L' in df.columns:
        rename_dict['PhotoPes_L'] = 'PhotoPes_likelihood'
    if 'W_PhotoPes' in df.columns:
        rename_dict['W_PhotoPes'] = 'WeightedPhotoPes'
    if 'W_PhotoPes_L' in df.columns:
        rename_dict['W_PhotoPes_L'] = 'W.PhotoPes_L'
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logging.info(f"列重命名完成: {rename_dict}")
    
    # 检查所有必需的列是否存在
    final_columns = ['news_date', 'PhotoPes', 'PhotoPes_likelihood', 'WeightedPhotoPes', 'W.PhotoPes_L']
    missing_columns = [col for col in final_columns if col not in df.columns]
    
    if missing_columns:
        logging.warning(f"Missing columns in DataFrame: {missing_columns}")
        # 为缺失的列添加默认值(NaN)
        for col in missing_columns:
            df[col] = float('nan')
    
    # 现在所有列都存在，可以安全地进行列选择
    df = df[final_columns]

    if output_file:
        # 确保目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved daily PhotoPes indicators to {output_file}")
        
    return df

def update_excel_with_photopes(df, excel_file):
    """
    更新Excel文件，添加所有PhotoPes指标
    
    参数:
        df: 包含PhotoPes指标的DataFrame
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
        merge_columns = ['Date', 'PhotoPes', 'PhotoPes_likelihood', 'WeightedPhotoPes', 'W.PhotoPes_L']
        
        # 检查列是否都存在，如果不存在则记录警告
        missing_cols = [col for col in merge_columns if col not in df_merge.columns]
        if missing_cols:
            logging.warning(f"合并Excel时缺少以下列: {missing_cols}")
            # 只选择存在的列
            available_columns = [col for col in merge_columns if col in df_merge.columns]
            df_merge = df_merge[available_columns]
            
        # 合并数据，如果列已存在则先删除
        existing_cols_to_drop = [col for col in merge_columns if col in df_excel.columns and col != 'Date']
        if existing_cols_to_drop:
            df_excel = df_excel.drop(columns=existing_cols_to_drop)
            
        df_updated = pd.merge(df_excel, df_merge, on="Date", how="left")
        
        # 保存更新后的Excel文件
        output_file = excel_file.replace(".xlsx", "_with_PhotoPes_Deduped.xlsx")
        df_updated.to_excel(output_file, index=False, engine='openpyxl')
        
        logging.info(f"已将所有PhotoPes指标添加到Excel文件: {output_file}")
        
        # 打印更新统计
        total_rows = len(df_excel)
        updated_rows = df_updated["WeightedPhotoPes"].notna().sum()
        logging.info(f"总行数: {total_rows}")
        logging.info(f"更新行数: {updated_rows}")
        logging.info(f"更新比例: {updated_rows/total_rows*100:.2f}%")
        
    except Exception as e:
        logging.error(f"更新Excel文件时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算每日加权PhotoPes指标 (去重版本)")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2026)),
                        help="要处理的年份列表")
    parser.add_argument("--output", type=str, default="data/weighted_photopes_deduped.csv",
                        help="输出文件路径")
    parser.add_argument("--update-excel", type=str, default=None,
                        help="要更新的Excel文件路径")
    
    args = parser.parse_args()
    
    # 连接数据库
    db = connect_to_mongodb()
    
    # 计算每日加权PhotoPes指标
    df = calculate_daily_photopes(db, args.years, args.output)
    
    # 更新Excel文件
    if args.update_excel and df is not None:
        update_excel_with_photopes(df, args.update_excel)
    
    logging.info("处理完成")

if __name__ == "__main__":
    main()
