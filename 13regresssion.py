#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为情绪指标对中国股市指数的回归结果生成LaTeX Beamer格式的表格。

该脚本会为每个市场指数生成一个独立的.tex文件，其中包含一个完整的Beamer框架，
其格式与用户提供的示例完全匹配。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import os

# --- 配置参数 ---
START_YEAR = 2014
END_YEAR = 2024
MAX_LAGS = 5
WINSORIZE_LEVEL = 0.01
OUTPUT_DIR = 'latex_tables'

# 定义情绪指标的映射（显示名称 -> (列名, 类型)）
SENTIMENT_INDICATORS = {
    'PhotoPes (Binary)': ('PhotoPes', 'Photo'),
    'PhotoPes Likelihood': ('PhotoPes_likelihood', 'Photo'),
    'W.PhotoPes (W.Bin)': ('WeightedPhotoPes', 'Photo'),
    'W.PhotoPes Likelihood': ('WeightedPhotoPes_likelihood', 'Photo'),
    'TextPes (Binary)': ('TextPes', 'Text'),
    'TextPes Likelihood': ('TextPes_likelihood', 'Text'),
    'W.TextPes (W.Bin)': ('WeightedTextPes', 'Text'),
    'W.TextPes Likelihood': ('WeightedTextPes_likelihood', 'Text'),
}

# 定义市场指数的映射 (完整名称 -> 简称)
MARKET_INDICES = {
    'CSI 300': 'CSI 300',
    'SSE Composite': 'SHCOMP',
    'SZSE Composite': 'SZCOMP',
    'ChiNext Composite': 'ChiNext',
    'CSI 500': 'CSI 500'
}

def get_significance_stars(p_value):
    """根据p值返回显著性星号"""
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    else:
        return ''

def format_coefficient(coef, p_value):
    """格式化系数和显著性星号"""
    stars = get_significance_stars(p_value)
    return f"{coef:.4f}{stars}"

def format_t_stat(t_stat):
    """格式化t值"""
    return f"({t_stat:.4f})"

def run_single_regression(df, sentiment_indicator, market_index):
    """对单个情绪指标和市场指数运行回归分析"""
    return_col = f"{market_index}_return"
    indicator_standardized = f"{sentiment_indicator}_standardized"

    # 创建滞后项
    for i in range(1, MAX_LAGS + 1):
        df[f'sentiment_lag_{i}'] = df[indicator_standardized].shift(i)
        df[f'return_lag_{i}'] = df[return_col].shift(i)
        df[f'return_sq_lag_{i}'] = (df[return_col]**2).shift(i)

    # 删除因创建滞后项产生的NaN值
    df_reg = df.dropna()

    # 定义回归变量
    X = df_reg[[
        f'sentiment_lag_{i}' for i in range(1, MAX_LAGS + 1)
    ] + [
        f'return_lag_{i}' for i in range(1, MAX_LAGS + 1)
    ] + [
        f'return_sq_lag_{i}' for i in range(1, MAX_LAGS + 1)
    ] + [
        'weekday_Tuesday', 'weekday_Wednesday', 'weekday_Thursday', 'weekday_Friday'
    ]]
    X = sm.add_constant(X)
    y = df_reg[return_col]

    # 运行OLS回归并使用HAC稳健标准误
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    return model, len(df_reg)

def generate_latex_for_index(market_index_full, market_index_short, df_full):
    """为单个市场指数生成完整的LaTeX Beamer框架"""
    print(f"--- Generating LaTeX for {market_index_full} ({market_index_short}) ---")
    
    table_rows = []
    sample_size = 0

    for i, (indicator_name, (col_name, indicator_type)) in enumerate(SENTIMENT_INDICATORS.items()):
        print(f"  Running regression for: {indicator_name}")
        
        # 为每个指标重新准备数据，以避免数据污染
        df = df_full.copy()
        df.dropna(subset=[col_name], inplace=True)
        
        # 缩尾和标准化
        indicator_winsorized = f"{col_name}_winsorized"
        indicator_standardized = f"{col_name}_standardized"
        df[indicator_winsorized] = winsorize(df[col_name].astype(float), limits=[WINSORIZE_LEVEL, WINSORIZE_LEVEL])
        df[indicator_standardized] = (df[indicator_winsorized] - df[indicator_winsorized].mean()) / df[indicator_winsorized].std()
        
        # 运行回归
        model, n_obs = run_single_regression(df, col_name, market_index_full)
        if i == 0: # 仅在第一次迭代时设置样本量
            sample_size = n_obs

        # 提取结果
        params = model.params
        pvalues = model.pvalues
        tvalues = model.tvalues

        # 提取滞后项结果
        lag_coefs = [format_coefficient(params[f'sentiment_lag_{j}'], pvalues[f'sentiment_lag_{j}']) for j in range(1, MAX_LAGS + 1)]
        lag_tvals = [format_t_stat(tvalues[f'sentiment_lag_{j}']) for j in range(1, MAX_LAGS + 1)]

        # 计算系数和
        sum_3_5_coef = params['sentiment_lag_3'] + params['sentiment_lag_4'] + params['sentiment_lag_5']
        sum_4_5_coef = params['sentiment_lag_4'] + params['sentiment_lag_5']
        
        # 使用t_test进行假设检验以获得和的t值和p值
        t_test_3_5 = model.t_test('sentiment_lag_3 + sentiment_lag_4 + sentiment_lag_5 = 0')
        t_test_4_5 = model.t_test('sentiment_lag_4 + sentiment_lag_5 = 0')

        sum_3_5_str = f"{format_coefficient(sum_3_5_coef, t_test_3_5.pvalue)}"
        sum_3_5_tval_str = f"{format_t_stat(t_test_3_5.tvalue.item())}"
        sum_4_5_str = f"{format_coefficient(sum_4_5_coef, t_test_4_5.pvalue)}"
        sum_4_5_tval_str = f"{format_t_stat(t_test_4_5.tvalue.item())}"
        
        # R2 和 Adj. R2
        r2 = f"{model.rsquared:.4f}"
        adj_r2 = f"{model.rsquared_adj:.4f}"
        
        # 加粗指定的行
        row_format = "\\textbf{{{}}}" if "Likelihood" in indicator_name else "{}"
        
        # 构建两行LaTeX表格代码
        coef_row = f"{row_format.format(indicator_name)} & {row_format.format(lag_coefs[0])} & {row_format.format(lag_coefs[1])} & {row_format.format(lag_coefs[2])} & {row_format.format(lag_coefs[3])} & {row_format.format(lag_coefs[4])} & {row_format.format(sum_3_5_str)} & {row_format.format(sum_4_5_str)} & {row_format.format(r2)} & {row_format.format(adj_r2)} \\"
        tval_row = f"                    & {row_format.format(lag_tvals[0])} & {row_format.format(lag_tvals[1])} & {row_format.format(lag_tvals[2])} & {row_format.format(lag_tvals[3])} & {row_format.format(lag_tvals[4])} & {row_format.format(sum_3_5_tval_str)} & {row_format.format(sum_4_5_tval_str)} \\"
        
        table_rows.append(coef_row)
        table_rows.append(tval_row)

        # 在照片和文本指标之间添加分隔线
        if indicator_name == 'W.PhotoPes Likelihood':
            table_rows.append('  \\midrule \\midrule')

    # 组合成完整的LaTeX文件内容
    table_content = "\n".join(table_rows)
    latex_string = f"""\
\begin{{frame}}[fragile]{{回归结果：{market_index_short}}}
    \\frametitle{{情感指标对{market_index_short}收益率的影响 (OLS)}}
\\begin{{adjustbox}}{{max width=\\textwidth, max height=0.8\\textheight}}
\\tiny
\\begin{{tabular}}{{lccccc|cc|cc}}
  \\toprule
  \\textbf{{指标 ({market_index_short})}} & \\textbf{{Lag 1}} & \\textbf{{Lag 2}} & \\textbf{{Lag 3}} & \\textbf{{Lag 4}} & \\textbf{{Lag 5}} & \\textbf{{Sum(t-3..5)}} & \\textbf{{Sum(t-4..5)}} & \\textbf{{$R^2$}} & \\textbf{{Adj.$R^2$}} \\
  & ($\beta$, t) & ($\beta$, t) & ($\beta$, t) & ($\beta$, t) & ($\beta$, t) & (Coef, t) & (Coef, t) & & \\
  \\midrule
{table_content}
  \\bottomrule
  \\multicolumn{{10}}{{l}}{{\\textit{{注：N = {sample_size}。W.Bin = Weighted Binary。显著性: * p<0.1, ** p<0.05, *** p<0.01。}}}}
\\end{{tabular}}
\\end{{adjustbox}}
\\end{{frame}}
"""

    # 保存到文件
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, f'regression_table_{market_index_short}.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_string)
    print(f"  Successfully saved LaTeX table to: {output_path}")

def main():
    """主函数：加载数据并为所有指数生成表格"""
    try:
        df = pd.read_csv('data/merged_sentiment_and_returns.csv', parse_dates=['news_date'])
    except FileNotFoundError:
        print("错误: 未找到 'data/merged_sentiment_and_returns.csv'。")
        print("请确保数据文件位于 'data' 目录中。")
        return

    # 预处理数据
    df['year'] = pd.DatetimeIndex(df['news_date']).year
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    
    all_sentiment_cols = [val[0] for val in SENTIMENT_INDICATORS.values()]
    all_return_cols = [f"{idx}_return" for idx in MARKET_INDICES.keys()]
    
    for col in all_sentiment_cols + all_return_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. 删除任何股指收益率缺失的行
    df.dropna(subset=all_return_cols, how='any', inplace=True)
    print(f"数据预处理完成。分析期间: {START_YEAR}-{END_YEAR}。初始观测数: {len(df)}")

    # 2. 创建星期虚拟变量 (Monday为基准)
    df['weekday'] = df['news_date'].dt.day_name()
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday', dtype=int)
    df = pd.concat([df, weekday_dummies], axis=1)
    if 'weekday_Monday' in df.columns:
        df.drop('weekday_Monday', axis=1, inplace=True)

    # 为每个市场指数生成一个LaTeX文件
    for full_name, short_name in MARKET_INDICES.items():
        generate_latex_for_index(full_name, short_name, df)

if __name__ == "__main__":
    main()
