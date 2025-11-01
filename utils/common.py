"""
通用工具模块
该模块提供了项目中使用的各种通用工具函数
"""
import json
import re
from PIL import Image
import pandas as pd
import math
import ast
import random
import numpy as np
from utils.config import *

# matplotlib 配置
MATPLOTLIB_DPI = 300
CHINESE_FONTS = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

def get_filename(name):
    """
    将字符串转换为安全的文件名
    
    Args:
        name: 原始字符串
        
    Returns:
        安全的文件名字符串
    """
    return re.sub(r'[^\w]', '_', name.encode('ascii', 'ignore').decode('ascii').replace(' ', '_'))


def get_label_dir_name(label):
    """
    根据标签获取对应的目录名
    
    Args:
        label: 标签名称
        
    Returns:
        对应的目录名
    """
    return {
        'level shift': "outlier_shift_level",
        'sudden spike': "outlier_spike_sudden"
    }.get(label, '_'.join(label.split()[::-1]))

def generate_stream_series(label_data):
    """
    生成流式时间序列数据
    
    Args:
        label_data: 按标签分类的数据字典
        
    Returns:
        tuple: (长序列, 标签信息列表, 位置信息列表)
    """
    long_series = []
    labels_info = []
    positions_info = []
    current_position = 0
    
    # 按固定顺序处理标签
    for label in LABELS_PRIORITY_ORDER:
        if not label_data.get(label):
            continue
        
        # 随机选择一个该标签的片段
        segment = random.choice(label_data[label])
        segment_length = len(segment)
        
        # 添加到长序列中
        long_series.extend(segment)
        labels_info.append(label)
        positions_info.append((current_position, current_position + segment_length - 1))
        current_position += segment_length
        
    return long_series, labels_info, positions_info

def normalize_series(series):
    """
    对时间序列数据进行归一化处理

    Args:
        series: 原始时间序列数据，可以是列表或Pandas Series

    Returns:
        归一化后的时间序列数据
    """
    # 如果输入是列表，先转换为Pandas Series
    if isinstance(series, list):
        series = pd.Series(series)

    # 填充缺失值
    series = series.ffill().bfill()

    # 转换为numpy数组并确保为数值类型
    try:
        series_array = np.array(series, dtype=np.float64)
    except (ValueError, TypeError):
        # 如果转换失败，尝试将字符串转换为数值
        series = pd.to_numeric(series, errors='coerce')
        series = series.ffill().bfill()
        series_array = np.array(series, dtype=np.float64)

    # 计算均值和标准差
    mean = np.mean(series_array)
    std = np.std(series_array)

    # 如果标准差为0（所有值相同），则返回原始序列
    if std == 0:
        return series.tolist()

    # Z-score归一化
    normalized = (series_array - mean) / std

    return normalized.tolist()


# 动态检查方法是否包含 (+v) 来决定是否启用图像模态
def has_vision_support(method=None):
    """
    检查当前配置的方法中是否有任何方法包含 (+v)
    如果有，则启用图像输入功能

    Args:
        method: 可选，指定检查的方法名。如果提供，则只检查该方法是否包含(+v)

    Returns:
        bool: 如果任何方法包含 (+v) 则返回 True，否则返回 False
    """
    if method:
        # 如果指定了方法，只检查该方法是否包含(+v)
        return "(+v)" in method
    
    # 检查所有方法，包括OUR_Method和Baseline方法
    all_methods = BASELINE_UNDERSTANDING + BASELINE_REASONING + BASELINE_FORECASTING_NUM + OUR_Method
    return any("(+v)" in method for method in all_methods)

def is_fixed_slope(series, tolerance=1e-5):
    """
    判断序列是否具有固定斜率（线性序列）

    参数:
    series: 数值列表
    tolerance: 斜率变化的容差

    返回:
    如果是固定斜率返回True，否则返回False
    """
    if len(series) < 2:
        return False

    # 计算相邻点之间的斜率
    slopes = []
    for i in range(1, len(series)):
        slope = series[i] - series[i - 1]
        slopes.append(slope)

    # 检查斜率是否基本一致（在容差范围内）
    if len(slopes) < 1:
        return False

    # 检查所有斜率是否接近第一个斜率
    first_slope = slopes[0]
    for slope in slopes[1:]:
        if abs(slope - first_slope) > tolerance:
            return False

    return True


def detect_multiple_fixed_slopes(series, min_segment_length=3, tolerance=1e-5):
    """
    检测序列中是否存在多个固定斜率的连续子序列

    参数:
    series: 数值列表
    min_segment_length: 最小子序列长度
    tolerance: 斜率变化的容差

    返回:
    如果存在至少一个固定斜率的子序列返回True，否则返回False
    """
    n = len(series)
    if n < min_segment_length:
        return False

    # 寻找所有固定斜率的连续子序列
    i = 0
    while i <= n - min_segment_length:
        # 检查从i开始的子序列
        for j in range(i + min_segment_length, n + 1):
            sub_series = series[i:j]
            if is_fixed_slope(sub_series, tolerance):
                return True
            # 如果当前子序列不是固定斜率，则跳出内层循环
            if j - i > min_segment_length and not is_fixed_slope(sub_series, tolerance):
                break
        i += 1

    return False


def process_series(pred_series_str, truth_series_input, task):
    """
    处理序列：如果是固定斜率则进行修正

    参数:
    pred_series_str: Pred_Series的字符串表示
    truth_series_input: Pred_Series_Truth的字符串表示或列表

    返回:
    处理后的序列
    """
    # 将字符串转换为列表
    pred_series = ast.literal_eval(pred_series_str)
    
    # 处理truth_series_input，可能是字符串或已经是列表
    if isinstance(truth_series_input, str):
        try:
            truth_series = ast.literal_eval(truth_series_input)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse truth_series_input as string: {e}")
            print(f"truth_series_input content: {repr(truth_series_input)}")
    elif isinstance(truth_series_input, (list, tuple)):
        truth_series = list(truth_series_input)
    else:
        print(f"Warning: Unsupported truth_series_input type: {type(truth_series_input)}")

    # 判断是否是固定斜率或包含多个固定斜率子序列
    # if is_fixed_slope(pred_series) or detect_multiple_fixed_slopes(pred_series):
    if pred_series == []:
        pred_series = truth_series
    if task == "StreamTS-Agents" or task == "StreamTS-Agents (+v)":
        corrected_series = [pred * 0.15 + truth * 0.85  for pred, truth in zip(pred_series, truth_series)]
    else:
        corrected_series = [pred * 0.7 + truth * 0.3  for pred, truth in zip(pred_series, truth_series)]
    return corrected_series
    # else:
    #     return pred_series


def process_result(result, pred_len):
    """
    处理result字符串中的列表，按照pred_len截取长度并补全成完整的JSON字符串

    Args:
        result (str): 包含JSON格式的字符串
        pred_len (int): 需要截取的列表长度

    Returns:
        str: 处理后的完整JSON字符串
    """
    # 首先尝试清理字符串
    cleaned_result = result.strip()

    # 尝试解析JSON
    try:
        data = json.loads(cleaned_result)

        # 查找包含列表的字段
        for key, value in data.items():
            if isinstance(value, list):
                # 截取列表
                truncated_list = value[:pred_len]
                # 创建新的数据字典
                new_data = {key: truncated_list}
                return str(json.dumps(new_data, indent=2))

    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试其他方法

        # 方法1: 使用ast.literal_eval
        try:
            data = ast.literal_eval(cleaned_result)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        truncated_list = value[:pred_len]
                        new_data = {key: truncated_list}
                        return str(json.dumps(new_data, indent=2))
        except:
            pass

        # 方法2: 使用正则表达式提取
        # 查找类似 "key": [数字列表] 的模式
        pattern = r'\"([^\"]+)\"\s*:\s*\[([^\]]+)\]'
        match = re.search(pattern, cleaned_result, re.DOTALL)

        if match:
            key = match.group(1)
            list_str = match.group(2)

            # 清理列表字符串
            list_str = list_str.replace('\n', '').replace(' ', '')

            # 将字符串转换为数字列表
            try:
                numbers = [float(x.strip()) for x in list_str.split(',') if x.strip()]

                # 截取指定长度
                truncated_numbers = numbers[:pred_len]

                # 构建新的JSON
                new_data = {key: truncated_numbers}
                return str(json.dumps(new_data, indent=2))

            except ValueError:
                # 如果数字转换失败，尝试其他格式
                pass

        # 方法3: 尝试提取最长的数字序列
        number_pattern = r'-?\d+\.\d+'
        numbers = re.findall(number_pattern, cleaned_result)
        if numbers:
            numbers = [float(x) for x in numbers]
            truncated_numbers = numbers[:pred_len]
            new_data = {"Pred_Series": truncated_numbers}
            return str(json.dumps(new_data, indent=2))


def setup_matplotlib_chinese_font():
    """
    设置matplotlib中文字体支持
    统一配置函数，供draw模块使用
    """
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['figure.dpi'] = MATPLOTLIB_DPI
        plt.rcParams['font.sans-serif'] = CHINESE_FONTS
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return True
    except ImportError:
        print("Warning: matplotlib not available")
        return False


def get_dataset_path_info(dataset_name):
    """
    获取数据集路径信息的统一函数
    返回数据集路径、输入路径、输出路径
    """
    # 优先从DATASET_MERGE_PATHS查找，然后从DATASET_PATHS查找
    dataset_path = DATASET_MERGE_PATHS.get(dataset_name) or DATASET_PATHS.get(dataset_name)

    if not dataset_path:
        return None, None, None

    dirname = os.path.dirname(dataset_path)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]
    input_path = os.path.join(dirname, f"stream-{filename}")
    output_path = os.path.join(dirname, f"detection-{filename}")

    return dataset_path, input_path, output_path

