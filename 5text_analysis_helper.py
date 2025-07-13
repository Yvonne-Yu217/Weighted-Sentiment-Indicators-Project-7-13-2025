import os
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import re
import glob
import random
from collections import Counter
from collections import defaultdict
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient

# 设置matplotlib使用中文字体
def set_chinese_font():
    import platform
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

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
    years_to_process = list(range(2014, 2025))  # 默认处理2014-2024年
    logging.info(f"使用默认年份配置: {years_to_process}")

# 全局配置
SAMPLE_RATIO = 0.1  # 抽取比例10%
MIN_TEXT_LENGTH = 0  # 最小文本长度，用于过滤噪声
NUM_WORKERS = 4  # 并行处理的工作线程数
DB_HOST = "localhost"
DB_PORT = 27017
DB_NAME = "sina_news_dataset_test"

class TextAnalyzer:
    """图片文字分析器"""
    
    def __init__(self, images_dir='images'):
        """初始化分析器"""
        self.images_dir = images_dir
        self.text_lengths = []
        self.image_paths = []
        self.text_contents = []
        # 连接MongoDB
        self.client = MongoClient(DB_HOST, DB_PORT)
        self.db = self.client[DB_NAME]
        
    def find_filtered_images(self):
        """从数据库中获取通过基础筛选的图片路径"""
        logging.info("从数据库中获取通过基础筛选的图片路径...")
        
        all_filtered_images = []
        
        for year in years_to_process:
            filtered_collection_name = f"{year}_filtered"
            
            # 检查集合是否存在
            if filtered_collection_name not in self.db.list_collection_names():
                logging.warning(f"过滤集合 {filtered_collection_name} 不存在，跳过处理")
                continue
                
            filtered_collection = self.db[filtered_collection_name]
            
            # 查询通过基础筛选且有有效图片的记录
            filtered_docs = filtered_collection.find({
                "basic_filtered": True,
                "has_valid_images": True
            })
            
            # 收集所有有效图片的路径
            for doc in filtered_docs:
                valid_images = doc.get("valid_images", [])
                for img_info in valid_images:
                    abs_path = img_info.get("abs_path")
                    if abs_path and os.path.exists(abs_path):
                        all_filtered_images.append(abs_path)
            
        logging.info(f"共找到 {len(all_filtered_images)} 张通过基础筛选的图片")
        return all_filtered_images
        
    def stratified_sample_by_date(self, all_images, sample_ratio=0.1):
        """
        按日期分层采样，总体采样比例为sample_ratio，最终抽样总数为总图片数*sample_ratio
        """
        date_to_images = defaultdict(list)
        for img_path in all_images:
            parts = img_path.split(os.sep)
            for part in parts:
                if re.match(r'\d{4}-\d{2}-\d{2}', part):
                    date_to_images[part].append(img_path)
                    break
        total_images = len(all_images)
        target_sample_size = int(total_images * sample_ratio)
        date_keys = list(date_to_images.keys())
        date_counts = [len(date_to_images[d]) for d in date_keys]
        # 按比例分配每个日期的采样数
        raw_allocations = [count / total_images * target_sample_size for count in date_counts]
        allocations = [int(round(x)) for x in raw_allocations]
        # 调整 allocations 使总和等于 target_sample_size
        diff = target_sample_size - sum(allocations)
        while diff != 0:
            for i in range(len(allocations)):
                if diff == 0:
                    break
                # 增加或减少分配
                if diff > 0 and allocations[i] < len(date_to_images[date_keys[i]]):
                    allocations[i] += 1
                    diff -= 1
                elif diff < 0 and allocations[i] > 0:
                    allocations[i] -= 1
                    diff += 1
        # 每个分层内随机采样
        sampled = []
        for date, alloc in zip(date_keys, allocations):
            imgs = date_to_images[date]
            if len(imgs) <= alloc:
                sampled.extend(imgs)
            else:
                sampled.extend(random.sample(imgs, alloc))
        return sampled
        
    def sample_images(self, all_images, sample_ratio=SAMPLE_RATIO):
        """按比例随机采样图片"""
        sample_size = int(len(all_images) * sample_ratio)
        if len(all_images) <= sample_size:
            logging.info(f"图片总数 {len(all_images)} 小于采样数 {sample_size}，使用所有图片")
            return all_images
        logging.info(f"从 {len(all_images)} 张图片中随机采样 {sample_size} 张进行分析")
        return random.sample(all_images, sample_size)
        
    def extract_text(self, image_path):
        """从图片中提取文字"""
        try:
            # 读取图片
            img = Image.open(image_path)
            
            # 使用pytesseract进行OCR
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            
            # 清理文本
            cleaned_text = ' '.join(text.split())
            
            # 关闭图片
            img.close()
            
            return cleaned_text
        except Exception as e:
            logging.debug(f"处理图片出错 {image_path}: {e}")
            return ""
            
    def analyze_images(self):
        """分析图片中的文字"""
        # 从数据库中获取通过基础筛选的图片
        all_images = self.find_filtered_images()
        
        if not all_images:
            logging.warning("没有找到通过基础筛选的图片，无法进行文字分析")
            return
            
        # 分层采样，总体采样比例为10%
        sample_images = self.stratified_sample_by_date(all_images, sample_ratio=SAMPLE_RATIO)
        self.text_lengths = []
        self.image_paths = []
        self.text_contents = []
        
        logging.info("开始提取图片文字...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            texts = list(tqdm(executor.map(self.extract_text, sample_images), total=len(sample_images), desc="提取文字"))
            
        for image_path, text in zip(sample_images, texts):
            if len(text) >= MIN_TEXT_LENGTH:
                self.text_lengths.append(len(text))
                self.image_paths.append(image_path)
                self.text_contents.append(text)
                
        logging.info(f"共分析了 {len(sample_images)} 张图片，其中 {len(self.text_lengths)} 张包含文字")
        
    def calculate_statistics(self):
        """计算文字长度的统计信息"""
        if not self.text_lengths:
            logging.warning("没有找到包含文字的图片")
            return {}
            
        stats = {
            "min": min(self.text_lengths),
            "max": max(self.text_lengths),
            "mean": np.mean(self.text_lengths),
            "median": np.median(self.text_lengths),
            "std": np.std(self.text_lengths),
            "percentiles": {
                "10%": np.percentile(self.text_lengths, 10),
                "25%": np.percentile(self.text_lengths, 25),
                "50%": np.percentile(self.text_lengths, 50),
                "75%": np.percentile(self.text_lengths, 75),
                "90%": np.percentile(self.text_lengths, 90),
                "95%": np.percentile(self.text_lengths, 95),
                "99%": np.percentile(self.text_lengths, 99),
            }
        }
        
        return stats
        
    def suggest_thresholds(self, stats):
        """根据统计信息建议阈值"""
        if not stats:
            return {}
            
        # 使用百分位数来设定阈值
        thresholds = {
            "low_text": stats["percentiles"]["25%"],  # 25%分位数
            "medium_text": stats["percentiles"]["75%"],  # 75%分位数
            "high_text": stats["percentiles"]["95%"],  # 95%分位数
        }
        
        return thresholds
        
    def plot_distribution(self):
        """绘制文字长度分布图（美化版）"""
        if not self.text_lengths:
            logging.warning("没有找到包含文字的图片，无法绘制分布图")
            return
        
        # 只关注99.5%分位数以内的数据，避免极端值影响美观
        x_limit = np.percentile(self.text_lengths, 99.5)
        filtered_lengths = [l for l in self.text_lengths if l <= x_limit]
        
        plt.figure(figsize=(12, 8))
        
        # 绘制直方图
        plt.subplot(211)
        plt.hist(filtered_lengths, bins=50, alpha=0.7, color='royalblue', edgecolor='black')
        plt.title('图片文字长度分布直方图')
        plt.xlabel('文字长度')
        plt.ylabel('图片数量')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, x_limit)
        
        # 绘制核密度估计
        plt.subplot(212)
        try:
            density = gaussian_kde(filtered_lengths)
            xs = np.linspace(0, x_limit, 300)
            plt.plot(xs, density(xs), 'r-', lw=2)
            plt.title('图片文字长度核密度估计')
            plt.xlabel('文字长度')
            plt.ylabel('密度')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, x_limit)
            
            # 在核密度图上标记潜在阈值
            stats = self.calculate_statistics()
            thresholds = self.suggest_thresholds(stats)
            for name, value in thresholds.items():
                if 0 < value <= x_limit:
                    plt.axvline(x=value, color='green', linestyle='--', alpha=0.7)
                    plt.text(value, density(value) * 1.1, f"{name}: {value:.0f}", rotation=90, verticalalignment='bottom')
        except Exception as e:
            logging.error(f"绘制核密度估计失败: {e}")
        
        plt.tight_layout()
        plt.savefig('text_length_distribution.png')
        logging.info("分布图已保存为 text_length_distribution.png")
        
    def show_examples(self, num_examples=5):
        """显示每个文字量级别的图片示例"""
        if not self.text_lengths or not self.image_paths:
            logging.warning("没有找到包含文字的图片，无法显示示例")
            return
            
        stats = self.calculate_statistics()
        thresholds = self.suggest_thresholds(stats)
        
        # 按文字长度将图片分类
        categories = {
            "无文字": [],
            "低文字量": [],
            "中等文字量": [],
            "高文字量": []
        }
        
        for i, length in enumerate(self.text_lengths):
            if length == 0:
                categories["无文字"].append((self.image_paths[i], length, self.text_contents[i]))
            elif length < thresholds["low_text"]:
                categories["低文字量"].append((self.image_paths[i], length, self.text_contents[i]))
            elif length < thresholds["medium_text"]:
                categories["中等文字量"].append((self.image_paths[i], length, self.text_contents[i]))
            else:
                categories["高文字量"].append((self.image_paths[i], length, self.text_contents[i]))
                
        # 显示每个类别的示例
        for category, images in categories.items():
            if not images:
                continue
                
            logging.info(f"\n=== {category} 示例 (共 {len(images)} 张图片) ===")
            
            # 随机选择示例
            samples = random.sample(images, min(num_examples, len(images)))
            
            for i, (path, length, text) in enumerate(samples, 1):
                logging.info(f"示例 {i}:")
                logging.info(f"路径: {path}")
                logging.info(f"文字长度: {length}")
                logging.info(f"文字内容: {text[:100]}..." if len(text) > 100 else text)
                logging.info("-" * 50)
                
    def suggest_continuous_weight_function(self):
        """建议连续的权重函数"""
        if not self.text_lengths:
            logging.warning("没有文字长度数据，无法建议权重函数")
            return {}
            
        stats = self.calculate_statistics()
        
        # 动态设置midpoint和steepness
        midpoint = stats["percentiles"]["95%"]  # 使用95%分位数
        range_90_95 = stats["percentiles"]["95%"] - stats["percentiles"]["90%"]
        k = 1.0  # 常数，可根据实验调整（0.5到2.0）
        if range_90_95 > 0:
            steepness = k / range_90_95
        else:
            steepness = 0.05  # 默认值，防止分母为0
        # 限制steepness范围
        steepness = max(0.01, min(steepness, 0.1))
        
        min_weight = 0.1
        max_weight = 1.0
        
        logging.info("\n=== 连续权重函数建议 ===")
        logging.info(f"建议使用逻辑函数进行连续权重映射:")
        logging.info(f"weight = max_weight - (max_weight - min_weight) / (1 + exp(-steepness * (text_length - midpoint)))")
        logging.info(f"其中:")
        logging.info(f"  midpoint = {midpoint:.0f}  # 95%分位数")
        logging.info(f"  steepness = {steepness:.5f}  # 根据90%-95%分位数范围计算")
        logging.info(f"  min_weight = {min_weight:.1f}")
        logging.info(f"  max_weight = {max_weight:.1f}")
        
        # 绘制权重函数曲线
        plt.figure(figsize=(10, 6))
        
        # 生成x轴数据点（使用对数刻度）
        if stats["max"] > 0:
            x_max = min(stats["max"], stats["percentiles"]["99%"] * 2)  # 限制最大值，避免极端值影响图表
            x = np.logspace(0, np.log10(x_max), 1000)
            x = np.insert(x, 0, 0)  # 添加0点
        else:
            x = np.linspace(0, 100, 1000)
        
        # 计算对应的权重（统一公式）
        y = [max_weight - (max_weight - min_weight) / (1 + np.exp(-steepness * (length - midpoint))) for length in x]
        
        plt.plot(x, y, 'b-', lw=2)
        plt.title('文字长度与权重的连续映射关系')
        plt.xlabel('文字长度')
        plt.ylabel('权重')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # 添加对数刻度
        
        # 标记关键点
        key_lengths = [0, stats["percentiles"]["25%"], stats["percentiles"]["50%"], 
                       stats["percentiles"]["75%"], stats["percentiles"]["90%"], 
                       stats["percentiles"]["95%"], stats["percentiles"]["99%"]]
        
        for length in key_lengths:
            if length > 0:  # 避免在0处绘制
                weight = max_weight - (max_weight - min_weight) / (1 + np.exp(-steepness * (length - midpoint)))
                plt.plot(length, weight, 'ro')
                plt.text(length, weight + 0.02, f"({int(length)}, {weight:.2f})")
            
        plt.axhline(y=min_weight, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=midpoint, color='gray', linestyle='--', alpha=0.5)
        
        plt.savefig('weight_function.png')
        logging.info("权重函数曲线已保存为 weight_function.png")
        
        return {
            "function_type": "logistic",
            "midpoint": midpoint,
            "steepness": steepness,
            "min_weight": min_weight,
            "max_weight": max_weight
        }

def generate_code_snippet(weight_params):
    """生成可用于导入到主程序的代码片段（统一公式，修正缩进与格式）"""
    if not weight_params:
        return ""
    code = (
        "# 连续权重计算函数\n"
        "import numpy as np\n\n"
        "def calculate_sentiment_weight(text_length):\n"
        "    \"\"\"根据文字长度计算情感权重\"\"\"\n"
        "    # 权重参数\n"
        "    midpoint = {midpoint:.1f}  # 95%分位数\n"
        "    steepness = {steepness:.6f}  # 根据90%-95%分位数范围计算\n"
        "    min_weight = {min_weight:.1f}  # 最小权重\n"
        "    max_weight = {max_weight:.1f}  # 最大权重\n"
        "    \n"
        "    # 使用统一逻辑函数计算权重\n"
        "    weight = max_weight - (max_weight - min_weight) / (1 + np.exp(-steepness * (text_length - midpoint)))\n"
        "    return weight\n"
    ).format(**weight_params)
    return code

def main():
    """主函数"""
    logging.info("=== 开始图片文字分析 ===")
    
    # 获取命令行参数
    if len(sys.argv) > 1:
        images_dir = sys.argv[1]
    else:
        images_dir = 'images'
    
    # 创建分析器
    analyzer = TextAnalyzer(images_dir)
    
    # 分析图片
    analyzer.analyze_images()
    
    # 计算统计信息
    stats = analyzer.calculate_statistics()
    
    # 输出统计信息
    logging.info("\n=== 文字长度统计 ===")
    logging.info(f"总样本数: {len(analyzer.text_lengths)}")
    logging.info(f"最小值: {stats.get('min')}")
    logging.info(f"最大值: {stats.get('max')}")
    logging.info(f"平均值: {stats.get('mean'):.2f}")
    logging.info(f"中位数: {stats.get('median'):.2f}")
    logging.info(f"标准差: {stats.get('std'):.2f}")
    
    logging.info("\n百分位数:")
    percentiles = stats.get('percentiles', {})
    for name, value in percentiles.items():
        logging.info(f"{name}: {value:.2f}")
    
    # 建议阈值
    thresholds = analyzer.suggest_thresholds(stats)
    
    logging.info("\n=== 建议阈值 ===")
    logging.info(f"低文字量阈值: {thresholds.get('low_text'):.0f}")
    logging.info(f"中等文字量阈值: {thresholds.get('medium_text'):.0f}")
    logging.info(f"高文字量阈值: {thresholds.get('high_text'):.0f}")
    
    # 绘制分布图
    analyzer.plot_distribution()
    
    # 显示示例
    analyzer.show_examples()
    
    # 建议连续权重函数
    weight_params = analyzer.suggest_continuous_weight_function()
    
    # 生成代码片段
    code_snippet = generate_code_snippet(weight_params)
    
    # 保存代码片段
    with open('weight_function_code.py', 'w') as f:
        f.write(code_snippet)
    
    logging.info("\n权重函数代码已保存到 weight_function_code.py")
    logging.info("=== 分析完成 ===")

if __name__ == "__main__":
    main()