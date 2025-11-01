import os
import sys
import shutil
import glob
from pathlib import Path
from utils.config import *

# 添加utils目录到Python路径以导入config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from utils.config import TASK, DATASET_UNDERSTANDING, DATASET_REASONING, DATASET_FORECASTING_NUM, DATASET_TO_MERGE, DATASET_FORECASTING_EVENT

def clean_datasets(patterns):
    """
    根据config.py中的TASK配置，只清理指定任务对应数据集目录中的detection-*, stream-*, predict-*文件夹及其内容
    """
    # 获取数据集目录路径
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')

    # 确保数据集目录存在
    if not os.path.exists(datasets_dir):
        print(f"Warning: {datasets_dir} directory does not exist")
        return

    # 根据TASK配置获取需要清理的数据集列表
    datasets_to_clean = set()
    
    for task in TASK:
        if task == "UNDERSTANDING":
            datasets_to_clean.update(DATASET_UNDERSTANDING)
        elif task == "REASONING":
            datasets_to_clean.update(DATASET_REASONING)
            # 当TASK包含REASONING时，也需要清理DATASET_TO_MERGE中的数据集
            datasets_to_clean.update(DATASET_TO_MERGE)
        elif task == "FORECASTING_NUM":
            datasets_to_clean.update(DATASET_FORECASTING_NUM)
        elif task == "FORECASTING_EVENT":
            datasets_to_clean.update(DATASET_FORECASTING_EVENT)
    
    print(f"Tasks: {TASK}")
    print(f"Datasets: {datasets_to_clean}")
    
    if not datasets_to_clean:
        print("No datasets to clean based on current TASK configuration")
        return

    # 删除匹配模式的文件夹，但只在指定的数据集目录中
    for pattern in patterns:
        # 使用Path.rglob递归查找以指定模式开头的所有目录
        matched_dirs = []
        if pattern.startswith('detection-'):
            matched_dirs = [str(d) for d in Path(datasets_dir).rglob('detection-*') if d.is_dir()]
        elif pattern.startswith('stream-'):
            matched_dirs = [str(d) for d in Path(datasets_dir).rglob('stream-*') if d.is_dir()]
        elif pattern.startswith('predict-'):
            matched_dirs = [str(d) for d in Path(datasets_dir).rglob('predict-*') if d.is_dir()]

        for dir_path in matched_dirs:
            # 检查该目录是否属于需要清理的数据集
            should_clean = False
            for dataset_name in datasets_to_clean:
                if dataset_name in dir_path:
                    should_clean = True
                    break
            
            if should_clean and os.path.isdir(dir_path):
                try:
                    # 删除目录及其所有内容
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")
                except Exception as e:
                    print(f"Error deleting {dir_path}: {str(e)}")

    print("Cleanup completed.\n")


if __name__ == "__main__":
    # 定义要删除的文件夹模式
    patterns = ['detection-*', 'stream-*', 'predict-*']
    clean_datasets(patterns)