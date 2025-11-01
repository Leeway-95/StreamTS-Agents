import sys
import os
from preprocess.dataset_MCQ2 import create_stream_summary
from utils.config import *

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from preprocess.dataset_multivariate import *
from preprocess.dataset_univariate import *
from preprocess.dataset_event import *


"""
将数据集转换为流数据的主程序
转换后的数据将保存在原始数据集目录下的stream-{dataset_name}子目录中
"""


def merge_stream_summaries(output_dir):
    
    # 存储所有数据集的DataFrame
    all_dfs = []
    
    # 读取每个数据集的stream_summary.csv文件
    for dataset_name in DATASET_TO_MERGE:
        # 获取数据集路径
        dataset_path = DATASET_PATHS[dataset_name]
        # 推断stream_summary.csv文件路径
        dirname = os.path.dirname(dataset_path)
        filename = os.path.splitext(os.path.basename(dataset_path))[0]
        csv_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")
        
        # 检查文件是否存在
        if os.path.exists(csv_path):
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
                # print(f"Read {csv_path}")
            except Exception as e:
                print(f"Read {csv_path} Error: {e}")
    
    # 合并所有DataFrame
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # 重新生成连续的索引，确保不重复
        merged_df['Index'] = range(1, len(merged_df) + 1)
        # 将Dataset列替换成QATS-4
        if 'Dataset' in merged_df.columns:
            merged_df['Dataset'] = 'QATS-4'
        return merged_df
    else:
        return None


if __name__ == '__main__':
    results = []
    dataset_name = ""
    for dataset_path in DATASET_PATHS.values():
        dataset_name = os.path.basename(dataset_path)
        # 处理单变量QA数据集
        if "REASONING" in TASK and ("AIOps" == dataset_name or "NAB" == dataset_name or "Oracle" == dataset_name or "WeatherQA" == dataset_name):
            # 从数据集路径中提取数据集名称
            dataset_name = [name for name, path in DATASET_PATHS.items() if path == dataset_path][0]
            results.append(process_uni_qa_dataset_on_task(dataset_name))
        if "REASONING" in TASK and ("MCQ2" == dataset_name):
            dataset_name = [name for name, path in DATASET_PATHS.items() if path == dataset_path][0]
            results.append(create_stream_summary(dataset_path))
        # 处理事件预测数据集
        elif "FORECASTING_EVENT" in TASK and any(dataset_name in DATASET_FORECASTING_EVENT for dataset_name, path in DATASET_PATHS.items() if path == dataset_path):
            results.append(process_event_dataset(dataset_path))
        # 处理多变量数据集
        elif "FORECASTING_NUM" in TASK and ("ETTm" in DATASET_FORECASTING_NUM or "Weather" in DATASET_FORECASTING_NUM) and ("ETTm" == dataset_name or "Weather" == dataset_name):
            results.append(process_multi_dataset(dataset_path))
        # 处理Gold数据集
        elif "FORECASTING_NUM" in TASK and "Gold" in DATASET_FORECASTING_NUM and ("Gold" == dataset_name):
            results.append(process_gold_dataset())
        # 处理TSQA数据集
        elif "UNDERSTANDING" in TASK and ("TSQA" == dataset_name):
            results.append(process_tsqa_on_label_dataset())

    # 打印处理摘要
    print("\nProcessing summary:")
    for res in results:
        print(f"- {res}")

    # 打印输出目录信息
    # print(f"\nOutput directories created at:")
    # for name, path in DATASET_PATHS.items():
    #     # 根据DATASET_PATHS自动推断输出路径
    #     dirname = os.path.dirname(path)
    #     filename = os.path.splitext(os.path.basename(path))[0]
    #     output_dir = os.path.join(dirname, f"stream-{filename}")
    #     if os.path.exists(output_dir):
    #         print(f"- {name}: {os.path.abspath(output_dir)}")

    if "REASONING" in TASK:
       # 创建QATS-4目录并合并stream_summary.csv文件
       #  print("Generating QATS-4 Merged streaming samples:")

        # 创建QATS-4目录
        qats4_dir = os.path.join(project_root, "datasets", "QATS-4")
        os.makedirs(qats4_dir, exist_ok=True)

        # 创建stream-QATS-4目录
        stream_qats4_dir = os.path.join(qats4_dir, "stream-QATS-4")
        os.makedirs(stream_qats4_dir, exist_ok=True)

        # 合并stream_summary.csv文件
        merged_df = merge_stream_summaries(stream_qats4_dir)

        if merged_df is not None:
            # 保存合并后的CSV文件
            merged_csv_path = os.path.join(stream_qats4_dir, "stream_summary.csv")
            merged_df.to_csv(merged_csv_path, index=False)
            # print(f"Merged {os.path.abspath(merged_csv_path)}")
            print(f"- Processed QATS-4: {len(merged_df)} series \n")
        else:
            print("The dataset file to be merged was not found")