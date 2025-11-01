"""
数据合成模块
包含生成合成时间序列数据和无模式数据的功能
"""

import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
from data_converter import extract_patterns_from_output

# 更新样本数据 - 使用新的input格式，不包含时间序列数值
input_str = """You are a time series pattern recognition expert. 
**Sudden Spike Outlier**: A brief, extreme deviation that sharply breaks the normal trajectory. One or several narrow, vertical peaks or drops appear abruptly and revert quickly to the prior baseline. Visually, it looks like thin, isolated spikes that stand out clearly from surrounding points. 
**Level Shift Outlier**: A sudden and sustained change to a new stable level with jumps or drops and then remains at that new horizontal position for a noticeable period.
**Upward Trend**: A smooth, persistent increase over time. The line slopes consistently upward, maintaining a general direction despite small local oscillations. The overall trajectory forms a clear rising trend. 
**Downward Trend**: A smooth, persistent decrease over time. The line slopes consistently downward, showing a gradual decline with minor short-term noise. The overall trajectory forms a clear falling trend. 
**Fixed Seasonality**: A repeating cyclical pattern with stable frequency and amplitude. The line forms symmetric, regular waves with consistent peaks and troughs across time. Each cycle is similar in height and duration. 
**Shifting Seasonality**: A repeating cyclical pattern with unstable frequency and amplitude. The cycles gradually stretch, compress, or grow/shrink in amplitude.
**Obvious Volatility**: A notable change in fluctuation intensity. The line becomes sharply more jagged, with dense, irregular oscillations and larger amplitude variations compared to previous segments. Visually, the signal looks “noisier” and more turbulent than before. 
**No Temporal Pattern**: No clear outlier, trend, seasonality, or volatility pattern is visible. The line appears stationary, with small. 
Return the result in **strict JSON format**, containing **only one** of the following eight labels: ["Sudden Spike Outlier", "Level Shift Outlier", "Fixed Seasonality", "Shifting Seasonality", "Upward Trend", "Downward Trend", "Obvious Volatility", "No Temporal Pattern"]. 
When multiple patterns coexist, resolve conflicts by the following priority order: "** Outlier" > "** Trend" > "** Seasonality" > "Obvious Volatility" > "No Temporal Pattern". 
**Task**: 
1.Analyze the given time series image (blue line plot) and classify it into **exactly one dominant temporal pattern label** based on the following definitions and distinguishing visual cues. Focus on the overall shape, continuity, and repetition of the line.
2.convert the **input** time series into a **caption** that clearly identifies the key temporal pattern(s) it exhibits. The **caption** should summarize the primary insight(s) from the data, emphasize the most relevant and distinct temporal behaviors, and consider incorporating **statistical indicators** such as mean, standard deviation, and percentage change. **caption** should integrate relevant **statistical indicators**.
Output strictly as: {"Label": "<one_of_the_above_labels>", "Caption": "<caption>"}."""

def generate_synthetic_data(original_data, num_samples, instruction, selected_patterns=None, strategy='depth', global_position=0):
    """
    基于原始数据生成合成的时间序列数据

    Args:
        original_data: 原始的alpaca格式数据列表
        num_samples: 要生成的样本数量
        instruction: 指令文本
        selected_patterns: 指定要生成的模式类型列表，如果为None则生成所有模式
        strategy: 数据生成策略，'depth'=每张图片只有一种模式，'breadth'=每张图片可叠加多种模式
        global_position: 全局位置计数器，用于计算range属性

    Returns:
        生成的合成数据列表和更新后的全局位置
    """
    synthetic_data = []

    # 按模式类型分组原始数据
    pattern_groups = {}
    for item in original_data:
        # 从字符串输出中提取模式信息
        if isinstance(item['output'], str):
            pattern_dict = extract_patterns_from_output(item['output'])
            pattern = pattern_dict['Primary_Pattern']
        else:
            pattern = item['output']['Primary_Pattern']

        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(item)

    print(f"原始数据模式分布: {[(k, len(v)) for k, v in pattern_groups.items()]}")

    # 如果指定了特定模式，则只使用这些模式
    if selected_patterns:
        # 过滤出指定的模式，排除"No Temporal Pattern"（它有单独的生成函数）
        available_patterns = [p for p in selected_patterns if p != "No Temporal Pattern" and p in pattern_groups]
        if not available_patterns:
            print("警告: 没有找到指定的模式类型在原始数据中，将使用所有可用模式")
            available_patterns = list(pattern_groups.keys())
    else:
        available_patterns = list(pattern_groups.keys())
    
    print(f"将生成的模式类型: {available_patterns}")
    print(f"使用策略: {strategy}")

    for i in tqdm(range(num_samples), desc="生成合成数据", unit="条"):
        if strategy == 'depth':
            # Depth策略：每张图片只有一种模式
            pattern_type = random.choice(available_patterns)
            base_sample = random.choice(pattern_groups[pattern_type])
            new_sample = deepcopy(base_sample)
            
        else:  # breadth策略
            # Breadth策略：每张图片可以叠加多种模式
            # 随机选择2-4种模式进行叠加
            num_patterns = random.randint(2, min(4, len(available_patterns)))
            selected_patterns_list = random.sample(available_patterns, num_patterns)
            
            # 选择主要模式作为基础样本
            primary_pattern = selected_patterns_list[0]
            base_sample = random.choice(pattern_groups[primary_pattern])
            new_sample = deepcopy(base_sample)
            
            # 更新输出标签为组合模式
            combined_label = " + ".join(selected_patterns_list)
            new_sample['output'] = f"{combined_label}<\\s>"

        # 生成随机长度的时间序列（模拟合成数据的长度）
        synthetic_length = random.randint(32, 512)  # 随机长度范围
        
        # 计算range属性：[start, end]
        range_start = global_position + 1
        range_end = global_position + synthetic_length
        range_info = [range_start, range_end]
        
        # 更新全局位置计数器
        global_position += synthetic_length

        # 更新样本数据（不再包含实际数值）
        new_sample['input'] = input_str
        new_sample['images'] = []  # 图片路径数组将在后续填充
        new_sample['range'] = range_info  # 添加range属性
        synthetic_data.append(new_sample)

    return synthetic_data, global_position


def generate_no_pattern_data(num_samples, instruction, global_position=0):
    """
    生成No Temporal Pattern的时序数据，参考其他数据的数值范围和精度

    Args:
        num_samples: 要生成的样本数量
        instruction: 指令文本
        global_position: 全局位置计数器，用于计算range属性

    Returns:
        生成的No Temporal Pattern数据列表和更新后的全局位置
    """
    no_pattern_data = []

    for i in tqdm(range(num_samples), desc="生成No Temporal Pattern数据", unit="条"):
        # 生成随机长度的时间序列（模拟无模式数据的长度）
        no_pattern_length = random.randint(32, 512)  # 随机长度范围
        
        # 计算range属性：[start, end]
        range_start = global_position + 1
        range_end = global_position + no_pattern_length
        range_info = [range_start, range_end]
        
        # 更新全局位置计数器
        global_position += no_pattern_length
        
        # 创建Alpaca格式的条目
        alpaca_item = {
            "instruction": instruction,
            "input": input_str,
            "output": "No Temporal Pattern<\\s>",
            "images": [],  # 图片路径数组将在后续填充
            "range": range_info  # 添加range属性
        }

        no_pattern_data.append(alpaca_item)

    return no_pattern_data, global_position


def generate_single_pattern_series(pattern_type):
    """
    生成单一模式的时间序列数据
    
    Args:
        pattern_type: 模式类型
    
    Returns:
        生成的时间序列数组
    """
    # 增加时间跨度多样性：生成长短不同的时间序列
    series_length = random.randint(30, 150)
    x = np.arange(series_length)
    
    # 根据模式类型生成相应的时间序列（复用原有逻辑）
    if 'Upward Trend' in pattern_type:
        return _generate_upward_trend_series(x, series_length)
    elif 'Downward Trend' in pattern_type:
        return _generate_downward_trend_series(x, series_length)
    elif 'Sudden Spike Outlier' in pattern_type:
        return _generate_sudden_spike_series(x, series_length)
    elif 'Level Shift Outlier' in pattern_type:
        return _generate_level_shift_series(x, series_length)
    elif 'Fixed Seasonality' in pattern_type:
        return _generate_fixed_seasonality_series(x, series_length)
    elif 'Shifting Seasonality' in pattern_type:
        return _generate_shifting_seasonality_series(x, series_length)
    elif 'Obvious Volatility' in pattern_type:
        return _generate_obvious_volatility_series(x, series_length)
    else:
        # 默认生成无模式数据
        base_level = np.random.uniform(-1, 1)
        noise_level = np.random.uniform(0.05, 0.15)
        return base_level + np.random.normal(0, noise_level, series_length)


def generate_combined_pattern_series(pattern_types):
    """
    生成组合多种模式的时间序列数据
    
    Args:
        pattern_types: 模式类型列表
    
    Returns:
        生成的组合时间序列数组
    """
    # 生成基础时间序列长度
    series_length = random.randint(30, 150)
    x = np.arange(series_length)
    
    # 初始化基础序列
    combined_series = np.zeros(series_length)
    
    # 为每种模式分配权重
    weights = np.random.dirichlet(np.ones(len(pattern_types)))
    
    for i, pattern_type in enumerate(pattern_types):
        # 生成单一模式序列
        single_series = generate_single_pattern_series(pattern_type)
        
        # 如果长度不匹配，进行调整
        if len(single_series) != series_length:
            if len(single_series) > series_length:
                single_series = single_series[:series_length]
            else:
                # 简单的线性插值扩展到目标长度
                x_old = np.linspace(0, 1, len(single_series))
                x_new = np.linspace(0, 1, series_length)
                single_series = np.interp(x_new, x_old, single_series)
        
        # 按权重叠加
        combined_series += weights[i] * single_series
    
    return combined_series


def _generate_upward_trend_series(x, series_length):
    """生成上升趋势时间序列"""
    trend_slope = np.random.uniform(0.005, 0.05)
    base_level = np.random.uniform(-2, 2)
    noise_level = np.random.uniform(0.02, 0.08)

    # 40%概率生成周期性上涨趋势
    if np.random.random() < 0.4:
        period = np.random.randint(15, 35)
        cycle_amplitude = np.random.uniform(0.2, 0.6)
        phase = np.random.uniform(0, 2 * np.pi)
        new_series = (base_level + trend_slope * x +
                      cycle_amplitude * np.sin(2 * np.pi * x / period + phase) +
                      np.random.normal(0, noise_level, series_length))
    elif np.random.random() < 0.3:
        # 非线性上升趋势
        new_series = base_level + trend_slope * x + 0.001 * x ** 1.5 + np.random.normal(0, noise_level, series_length)
    else:
        # 线性上升趋势
        new_series = base_level + trend_slope * x + np.random.normal(0, noise_level, series_length)
    
    return new_series


def _generate_downward_trend_series(x, series_length):
    """生成下降趋势时间序列"""
    trend_slope = np.random.uniform(-0.05, -0.005)
    base_level = np.random.uniform(-2, 2)
    noise_level = np.random.uniform(0.02, 0.08)

    # 40%概率生成周期性下降趋势
    if np.random.random() < 0.4:
        period = np.random.randint(15, 35)
        cycle_amplitude = np.random.uniform(0.2, 0.6)
        phase = np.random.uniform(0, 2 * np.pi)
        new_series = (base_level + trend_slope * x +
                      cycle_amplitude * np.sin(2 * np.pi * x / period + phase) +
                      np.random.normal(0, noise_level, series_length))
    elif np.random.random() < 0.3:
        # 非线性下降趋势
        new_series = base_level + trend_slope * x - 0.001 * x ** 1.5 + np.random.normal(0, noise_level, series_length)
    else:
        # 线性下降趋势
        new_series = base_level + trend_slope * x + np.random.normal(0, noise_level, series_length)
    
    return new_series


def _generate_sudden_spike_series(x, series_length):
    """生成突发异常时间序列"""
    base_level = np.random.uniform(-1, 1)
    noise_level = np.random.uniform(0.02, 0.08)
    
    # 40%概率生成有周期性背景的突发异常
    if np.random.random() < 0.4:
        period = np.random.randint(15, 30)
        amplitude = np.random.uniform(0.3, 0.8)
        phase = np.random.uniform(0, 2 * np.pi)
        new_series = base_level + amplitude * np.sin(2 * np.pi * x / period + phase) + np.random.normal(0, noise_level, series_length)
    else:
        # 纯随机背景
        new_series = base_level + np.random.normal(0, noise_level, series_length)

    num_spikes = np.random.randint(1, 4)
    for _ in range(num_spikes):
        min_pos = min(5, series_length // 6)
        max_pos = max(series_length - 5, series_length * 5 // 6)
        spike_pos = random.randint(min_pos, max_pos)

        spike_type = np.random.choice(['small', 'medium', 'large'], p=[0.6, 0.3, 0.1])
        if spike_type == 'small':
            spike_magnitude = random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        elif spike_type == 'medium':
            spike_magnitude = random.choice([-1, 1]) * np.random.uniform(1.5, 3.0)
        else:  # large
            spike_magnitude = random.choice([-1, 1]) * np.random.uniform(3.0, 5.0)

        spike_width = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])

        for i in range(spike_width):
            if spike_pos + i < series_length:
                decay_factor = np.exp(-i * 0.8)
                new_series[spike_pos + i] += spike_magnitude * decay_factor
    
    return new_series


def _generate_level_shift_series(x, series_length):
    """生成水平偏移异常时间序列"""
    base_level = np.random.uniform(-1, 1)
    noise_level = np.random.uniform(0.02, 0.08)
    
    # 40%概率生成有周期性背景的水平偏移异常
    if np.random.random() < 0.4:
        period = np.random.randint(15, 30)
        amplitude = np.random.uniform(0.3, 0.8)
        phase = np.random.uniform(0, 2 * np.pi)
        new_series = base_level + amplitude * np.sin(2 * np.pi * x / period + phase) + np.random.normal(0, noise_level, series_length)
    else:
        # 纯随机背景
        new_series = base_level + np.random.normal(0, noise_level, series_length)

    num_shifts = np.random.randint(1, 3)
    min_pos = min(10, series_length // 4)
    max_pos = max(series_length - 10, series_length * 3 // 4)

    if max_pos > min_pos:
        shift_positions = sorted(np.random.choice(range(min_pos, max_pos), num_shifts, replace=False))
    else:
        shift_positions = [series_length // 2]

    for shift_pos in shift_positions:
        shift_type = np.random.choice(['micro', 'small', 'medium', 'large'], p=[0.4, 0.35, 0.2, 0.05])
        if shift_type == 'micro':
            shift_magnitude = random.choice([-1, 1]) * np.random.uniform(0.2, 0.6)
        elif shift_type == 'small':
            shift_magnitude = random.choice([-1, 1]) * np.random.uniform(0.6, 1.2)
        elif shift_type == 'medium':
            shift_magnitude = random.choice([-1, 1]) * np.random.uniform(1.2, 2.0)
        else:  # large
            shift_magnitude = random.choice([-1, 1]) * np.random.uniform(2.0, 3.5)

        transition_length = np.random.randint(1, min(5, series_length - shift_pos))

        for i in range(transition_length):
            if shift_pos + i < series_length:
                transition_factor = 1 / (1 + np.exp(-3 * (i / transition_length - 0.5)))
                new_series[shift_pos + i] += shift_magnitude * transition_factor

        if shift_pos + transition_length < series_length:
            new_series[shift_pos + transition_length:] += shift_magnitude
    
    return new_series


def _generate_fixed_seasonality_series(x, series_length):
    """生成固定季节性时间序列"""
    base_level = np.random.uniform(-1, 1)
    max_period = series_length // 2
    period = random.randint(12, min(30, max_period))
    amplitude = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2 * np.pi)
    noise_level = np.random.uniform(0.01, 0.06)

    if np.random.random() < 0.5:
        new_series = base_level + amplitude * np.sin(2 * np.pi * x / period + phase)
    else:
        new_series = base_level + amplitude * np.cos(2 * np.pi * x / period + phase)

    if np.random.random() < 0.3:
        harmonic_amplitude = amplitude * np.random.uniform(0.2, 0.5)
        new_series += harmonic_amplitude * np.sin(4 * np.pi * x / period + phase)

    new_series += np.random.normal(0, noise_level, series_length)
    return new_series


def _generate_shifting_seasonality_series(x, series_length):
    """生成变化季节性时间序列"""
    base_level = np.random.uniform(-1, 1)
    max_period = series_length // 2
    period = random.randint(12, min(30, max_period))
    base_amplitude = np.random.uniform(0.3, 1.5)
    phase = np.random.uniform(0, 2 * np.pi)
    noise_level = np.random.uniform(0.01, 0.06)

    # 随机选择幅度变化方向：增大或减小
    amplitude_direction = np.random.choice(['increasing', 'decreasing'])

    if amplitude_direction == 'increasing':
        amplitude_change = np.random.uniform(0.5, 2.0)
        if np.random.random() < 0.5:
            amplitude = base_amplitude + amplitude_change * (x / series_length)
        else:
            amplitude = base_amplitude + amplitude_change * ((x / series_length) ** 1.5)
    else:
        amplitude_change = np.random.uniform(0.5, 2.0)
        if np.random.random() < 0.5:
            amplitude = base_amplitude + amplitude_change - amplitude_change * (x / series_length)
        else:
            amplitude = base_amplitude + amplitude_change - amplitude_change * ((x / series_length) ** 1.5)

    # 确保幅度不为负数
    amplitude = np.maximum(amplitude, 0.1)

    new_series = base_level + amplitude * np.sin(2 * np.pi * x / period + phase) + np.random.normal(0, noise_level, series_length)
    return new_series


def _generate_obvious_volatility_series(x, series_length):
    """生成明显波动性时间序列"""
    base_level = np.random.uniform(-1, 1)
    noise_level = np.random.uniform(0.2, 0.5)
    
    # 60%概率生成高波动随机数据，40%概率生成随机游走
    if np.random.random() < 0.6:
        # 高波动随机数据：围绕基准水平的大幅随机变化
        new_series = base_level + np.random.normal(0, noise_level, series_length)
    else:
        # 随机游走：较大的步长变化
        new_series = np.zeros(series_length)
        new_series[0] = base_level
        step_size = noise_level * 0.4
        for i in range(1, series_length):
            new_series[i] = new_series[i - 1] + np.random.normal(0, step_size)
    
    return new_series