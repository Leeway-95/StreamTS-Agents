#!/usr/bin/env python3

import csv
import pandas as pd
import json
import os
import sys
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.config import TSQA_SAMPLE_RAN_CNT, DATASET_PATHS, SAMPLE_SIZE, GEN_STREAM_CNT


def create_stream_summary(dataset_path):
    # 判断传入的是文件路径还是目录路径
    processed_count = 0
    if dataset_path.endswith('.csv'):
        # 如果是CSV文件路径
        input_path = dataset_path
        dataset_dir = os.path.dirname(dataset_path)
        dataset_name = os.path.basename(dataset_path).replace('.csv', '')
    else:
        # 如果是目录路径
        dataset_name = os.path.basename(dataset_path)
        dataset_dir = os.path.dirname(dataset_path)
        # 首先检查直接路径是否存在CSV文件
        direct_csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
        nested_csv_path = os.path.join(dataset_path, f"{dataset_name}.csv")

        if os.path.exists(direct_csv_path):
            input_path = direct_csv_path
        elif os.path.exists(nested_csv_path):
            input_path = nested_csv_path
            dataset_dir = dataset_path
        else:
            # 如果都不存在，使用原来的逻辑
            dataset_dir = dataset_path
            input_path = os.path.join(dataset_dir, f"{dataset_name}.csv")

    output_dir = os.path.join(dataset_dir, f"stream-{dataset_name}")
    output_path = os.path.join(output_dir, f'stream_summary.csv')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输出文件是否存在，如果不存在则创建基本CSV文件
    if not os.path.exists(output_path):
        # 创建基本CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Dataset', 'Variable', 'Positions', 'Labels', 'Series', 'Question'])

    df = pd.read_csv(input_path)
    random.seed(SAMPLE_SIZE)
    
    # 使用GEN_STREAM_CNT配置
    
    # 创建多个流记录
    stream_records = []
    processed_count = 0
    
    # 生成GEN_STREAM_CNT个流数据序列
    for stream_idx in range(GEN_STREAM_CNT):
        # 为每个流随机选择TSQA_SAMPLE_RAN_CNT行
        if len(df) < TSQA_SAMPLE_RAN_CNT:
            selected_indices = list(range(len(df)))
        else:
            # 使用不同的随机种子确保每个流都不同
            random.seed(SAMPLE_SIZE + stream_idx)
            selected_indices = sorted(random.sample(range(len(df)), TSQA_SAMPLE_RAN_CNT))
        
        # 准备合并数据
        positions = []
        labels = []
        all_series = []
        all_questions = []
        
        current_position = 0
        
        for idx in selected_indices:
            row = df.iloc[idx]
            
            # 解析Series数据（从字符串转换为列表）
            try:
                series_data = json.loads(row['Series'])
            except json.JSONDecodeError:
                continue
            
            # 获取当前行的size和label
            size = int(row['Size'])  # 确保是Python原生int类型
            label = int(row['Label'])  # 确保是Python原生int类型
            question = row['Question']
            
            # 计算position范围：从current_position到current_position + size - 1
            position_start = current_position
            position_end = current_position + size - 1
            positions.append((position_start, position_end))
            
            # 添加label、series和question
            labels.append(label)
            all_series.extend(series_data)
            all_questions.append(question)
            
            # 更新下一个position的起始点
            current_position = position_end + 1
        
        # 创建合并后的记录
        stream_record = {
            'Index': stream_idx + 1,  # 使用流索引
            'Dataset': 'MCQ2',  # 数据集名称
            'Variable': 'MCQ2',  # 变量名称
            'Positions': json.dumps(positions),  # position范围列表 [(start1, end1), (start2, end2), ...]
            'Labels': json.dumps(labels),  # labels列表 [label1, label2, ...]
            'Series': json.dumps(all_series),  # 合并的时间序列数据
            'Question': json.dumps(all_questions)  # 合并的问题数组
        }
        
        stream_records.append(stream_record)
        processed_count += 1
    
    # 创建DataFrame并保存
    stream_df = pd.DataFrame(stream_records)
    stream_df.to_csv(output_path, index=False)
    
    return f"Processed {dataset_name}: {processed_count} series"

if __name__ == "__main__":
    create_stream_summary()