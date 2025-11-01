import os

ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys
import ast
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

# 导入必要的模块
from clazz.segmentation import ClaSS
from clazz.window_size import suss
from clazz.profile import binary_f1_score

dataset_path = './'
def load_dataset(dataset, selection=None):
    desc_filename = ABS_PATH + f"{dataset_path}/{dataset}/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []

    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue
        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        path = ABS_PATH + f'{dataset_path}/{dataset}/'

        if os.path.exists(path + ts_name + ".txt"):
            ts = np.loadtxt(fname=path + ts_name + ".txt", dtype=np.float64)
        else:
            ts = np.load(file=path + "data.npz")[ts_name]

        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])


def load_train_dataset():
    train_names = [
        'DodgerLoopDay',
        'EEGRat',
        'EEGRat2',
        'FaceFour',
        'GrandMalSeizures2',
        'GreatBarbet1',
        'Herring',
        'InlineSkate',
        'InsectEPG1',
        'MelbournePedestrian',
        'NogunGun',
        'NonInvasiveFetalECGThorax1',
        'ShapesAll',
        'TiltECG',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'Trace',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'WordSynonyms',
        'Yoga'
    ]

    df = pd.concat([load_dataset("UTSA"), load_dataset("TSSB")])
    df = df[df["name"].isin(train_names)]

    return df.sort_values(by="name")


def load_benchmark_dataset():
    df = pd.concat([load_dataset("UTSA"), load_dataset("TSSB")])
    df.sort_values(by="name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_archives_dataset():
    df = pd.concat([
        load_dataset("PAMAP"),
        load_dataset("mHealth"),
        load_dataset("WESAD"),
        load_dataset("MIT-BIH-VE"),
        load_dataset("MIT-BIH-Arr"),
        load_dataset("SleepDB"),
    ])
    df.sort_values(by="name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_combined_dataset():
    df = pd.concat([
        load_benchmark_dataset(),
        load_archives_dataset()
    ])
    df.sort_values(by="name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_positions(positions_str: str) -> List[Tuple[int, int]]:
    """
    解析Positions列的字符串格式，转换为元组列表
    
    Args:
        positions_str: 形如 "[(0, 63), (64, 191), ...]" 的字符串
        
    Returns:
        List[Tuple[int, int]]: 分割位置的元组列表
    """
    try:
        # 使用ast.literal_eval安全地解析字符串
        positions = ast.literal_eval(positions_str)
        return positions
    except (ValueError, SyntaxError) as e:
        print(f"解析位置字符串时出错: {e}")
        return []


def parse_series(series_str: str) -> np.ndarray:
    """
    解析Series列的字符串格式，转换为numpy数组
    
    Args:
        series_str: 形如 "[0.073, 0.410, ...]" 的字符串
        
    Returns:
        np.ndarray: 时间序列数据
    """
    try:
        # 使用ast.literal_eval安全地解析字符串
        series_list = ast.literal_eval(series_str)
        return np.array(series_list, dtype=np.float64)
    except (ValueError, SyntaxError) as e:
        print(f"解析时间序列字符串时出错: {e}")
        return np.array([])


def segment_time_series(series: np.ndarray, return_profile=False, **kwargs) -> List[Tuple[int, int]]:
    """
    使用ClaSS算法对时间序列进行分割
    
    Args:
        series: 时间序列数据
        return_profile: 是否返回profile数据用于可视化
        **kwargs: ClaSS算法的参数
        
    Returns:
        List[Tuple[int, int]] 或 Tuple[List[Tuple[int, int]], np.ndarray]: 分割位置的元组列表，可选返回profile
    """
    if len(series) == 0:
        return []
    
    try:
        # 创建ClaSS分割器
        segmenter = ClaSS(
            n_timepoints=len(series),
            window_size=kwargs.get('window_size', suss),
            k_neighbours=kwargs.get('k_neighbours', 3),
            score=kwargs.get('score', binary_f1_score),
            jump=kwargs.get('jump', 5),
            p_value=kwargs.get('p_value', 1e-50),
            sample_size=kwargs.get('sample_size', 1000),
            similarity=kwargs.get('similarity', "pearson"),
            profile_mode=kwargs.get('profile_mode', "global"),
            verbose=kwargs.get('verbose', 0)
        )
        
        # 逐点更新时间序列
        for i, point in enumerate(series):
            segmenter.update(point)
        
        # 获取变化点
        change_points = segmenter.change_points
        
        # 将变化点转换为分割区间
        segments = []
        if len(change_points) == 0:
            # 如果没有变化点，整个序列作为一个分割
            segments = [(0, len(series) - 1)]
        else:
            # 添加起始点
            start = 0
            for cp in change_points:
                if cp > start:
                    segments.append((start, cp - 1))
                    start = cp
            
            # 添加最后一个分割
            if start < len(series):
                segments.append((start, len(series) - 1))
        
        if return_profile:
            # 获取有效的profile数据（去除-inf值）
            profile = segmenter.profile.copy()
            valid_mask = profile != -np.inf
            if np.any(valid_mask):
                return segments, profile
            else:
                # 如果没有有效profile，创建一个简单的profile
                simple_profile = np.zeros(len(series))
                return segments, simple_profile
        else:
            return segments
        
    except Exception as e:
        print(f"分割时间序列时出错: {e}")
        if return_profile:
            return [(0, len(series) - 1)], np.zeros(len(series))
        else:
            return [(0, len(series) - 1)]  # 返回整个序列作为一个分割


def calculate_coverage(predicted_segments: List[Tuple[int, int]], 
                      true_segments: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    计算分割结果的覆盖率
    
    Args:
        predicted_segments: 预测的分割结果
        true_segments: 真实的分割结果
        
    Returns:
        Dict[str, float]: 包含各种覆盖率指标的字典
    """
    if not predicted_segments or not true_segments:
        return {
            'intersection_coverage': 0.0,
            'union_coverage': 0.0,
            'jaccard_index': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # 创建覆盖集合
    predicted_points = set()
    true_points = set()
    
    for start, end in predicted_segments:
        predicted_points.update(range(start, end + 1))
    
    for start, end in true_segments:
        true_points.update(range(start, end + 1))
    
    # 计算交集和并集
    intersection = predicted_points.intersection(true_points)
    union = predicted_points.union(true_points)
    
    # 计算各种指标
    intersection_size = len(intersection)
    union_size = len(union)
    predicted_size = len(predicted_points)
    true_size = len(true_points)
    
    # 覆盖率指标
    intersection_coverage = intersection_size / true_size if true_size > 0 else 0.0
    union_coverage = intersection_size / union_size if union_size > 0 else 0.0
    jaccard_index = intersection_size / union_size if union_size > 0 else 0.0
    
    # 精确率、召回率和F1分数
    precision = intersection_size / predicted_size if predicted_size > 0 else 0.0
    recall = intersection_size / true_size if true_size > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'intersection_coverage': intersection_coverage,
        'union_coverage': union_coverage,
        'jaccard_index': jaccard_index,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def process_dataset(csv_path: str = f'{dataset_path}/stream_summary.csv') -> pd.DataFrame:
    """
    处理整个数据集，对每行数据进行分割和覆盖率计算
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        pd.DataFrame: 包含原始数据和计算结果的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")
    
    # 读取CSV文件
    print(f"正在读取数据文件: {csv_path}")
    df = pd.read_csv(csv_path, sep=',')
    
    # 检查必要的列是否存在
    required_columns = ['Series', 'Positions']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    # 初始化结果列
    df['Predicted_Segments'] = None
    df['Coverage_Metrics'] = None
    
    print(f"开始处理 {len(df)} 行数据...")
    
    # 处理每一行数据
    for idx, row in df.iterrows():
        print(f"处理第 {idx + 1}/{len(df)} 行数据...")
        
        try:
            # 解析时间序列数据
            series = parse_series(row['Series'])
            if len(series) == 0:
                print(f"第 {idx + 1} 行: 时间序列数据为空，跳过")
                continue
            
            # 解析真实分割位置
            true_segments = parse_positions(row['Positions'])
            if len(true_segments) == 0:
                print(f"第 {idx + 1} 行: 真实分割位置为空，跳过")
                continue
            
            print(f"  时间序列长度: {len(series)}")
            print(f"  真实分割数量: {len(true_segments)}")
            print(f"  真实分割: {true_segments}")
            
            # 进行时间序列分割
            predicted_segments = segment_time_series(series)
            print(f"  预测分割数量: {len(predicted_segments)}")
            print(f"  预测分割: {predicted_segments}")
            
            # 计算覆盖率
            coverage_metrics = calculate_coverage(predicted_segments, true_segments)
            
            # 保存结果
            df.at[idx, 'Predicted_Segments'] = str(predicted_segments)
            df.at[idx, 'Coverage_Metrics'] = str(coverage_metrics)
            
            # 打印覆盖率指标
            print(f"  覆盖率指标:")
            for metric, value in coverage_metrics.items():
                print(f"    {metric}: {value:.4f}")
            print()
            
        except Exception as e:
            print(f"处理第 {idx + 1} 行数据时出错: {e}")
            continue
    
    return df


def create_results_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建结果CSV文件，只包含Index、Dataset、Predict_Positions、Size四列
    
    Args:
        df: 包含完整数据的DataFrame
        
    Returns:
        pd.DataFrame: 只包含指定四列的DataFrame
    """
    results_data = []
    
    for idx, row in df.iterrows():
        try:
            # 解析预测的分割位置
            if pd.notna(row['Predicted_Segments']):
                predicted_segments_str = str(row['Predicted_Segments'])
                
                # 尝试多种方式解析字符串
                predicted_segments = None
                try:
                    # 首先尝试 ast.literal_eval
                    predicted_segments = ast.literal_eval(predicted_segments_str)
                except (ValueError, SyntaxError):
                    try:
                        # 如果失败，尝试 eval（注意：这里只处理我们知道安全的数据）
                        predicted_segments = eval(predicted_segments_str)
                    except:
                        # 如果都失败，跳过这一行
                        print(f"无法解析第 {idx} 行的预测分割数据: {predicted_segments_str}")
                        continue
                
                if predicted_segments is not None:
                    # 将numpy数据类型转换为普通Python整数，并计算每个预测范围的长度
                    clean_segments = []
                    sizes = []
                    for start, end in predicted_segments:
                        # 确保转换为普通Python整数
                        start_int = int(start)
                        end_int = int(end)
                        clean_segments.append((start_int, end_int))
                        
                        size = end_int - start_int + 1
                        sizes.append(size)
                    
                    results_data.append({
                        'Index': int(row['Index']) if 'Index' in row else idx,
                        'Dataset': str(row['Dataset']) if 'Dataset' in row else 'Unknown',
                        'Predict_Positions': str(clean_segments),
                        'Size': str(sizes)
                    })
            
        except Exception as e:
            print(f"处理第 {idx} 行数据时出错: {e}")
            continue
    
    return pd.DataFrame(results_data)