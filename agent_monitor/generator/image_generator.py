"""
图像生成模块
包含时间序列图表绘制和图像生成相关的功能
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import ast
from tqdm import tqdm
from data_converter import extract_patterns_from_output, get_pattern_folder_name
from folder_manager import create_pattern_folders, create_samples_folder, create_no_pattern_samples_folder


def plot_time_series(series_data, output_label, index, save_path, show_pattern_label=True):
    """
    绘制时间序列图并添加标签信息

    Args:
        series_data: 时间序列数据列表
        output_label: 输出标签字典，包含Primary_Pattern和Sub_Pattern
        index: 序列索引（从1开始）
        save_path: 保存路径
        show_pattern_label: 是否在标题中显示模式标签
    """
    plt.figure(figsize=(8, 6))

    # 绘制时间序列
    x = np.arange(len(series_data))

    # 根据模式类型设置不同的颜色 - 七种时序模式分类都不一样，no_temporal_pattern用灰色
    # color_map = {
    #     'Upward Trend': '#267BB6',
    #     'Downward Trend': 'navy',
    #     'Sudden Spike Outlier': 'red',
    #     'Level Shift Outlier': 'orange',
    #     'Fixed Seasonality': 'green',
    #     'Shifting Seasonality': 'purple',
    #     'Obvious Volatility': 'darkred',
    #     'No Temporal Pattern': 'gray',
    #     'Unknown': 'gray'
    # }

    color_map = {
        'Upward Trend': '#267BB6',
        'Downward Trend': '#267BB6',
        'Sudden Spike Outlier': '#267BB6',
        'Level Shift Outlier': '#267BB6',
        'Fixed Seasonality': '#267BB6',
        'Shifting Seasonality': '#267BB6',
        'Obvious Volatility': '#267BB6',
        'No Temporal Pattern': '#267BB6',
        'Unknown': 'gray'
    }


    # 如果output_label是字符串，需要先解析
    if isinstance(output_label, str):
        pattern_dict = extract_patterns_from_output(output_label)
        primary_pattern = pattern_dict['Primary_Pattern']
        sub_pattern = pattern_dict['Sub_Pattern']
    else:
        primary_pattern = output_label.get('Primary_Pattern', 'Unknown')
        sub_pattern = output_label.get('Sub_Pattern', 'Unknown')

    line_color = color_map.get(primary_pattern, '#267BB6')

    plt.plot(x, series_data, color=line_color, linewidth=6, alpha=0.8)

    # 去掉网格和坐标系
    plt.axis('off')

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存


def generate_sample_images(synthetic_data, start_index):
    """
    为合成数据每隔20张生成采样图片

    Args:
        synthetic_data: 合成数据列表
        start_index: 起始索引（原始数据的数量）
    """
    create_samples_folder()

    sample_count = 0
    for i in range(0, len(synthetic_data), 20):  # 每隔20张采样一次
        data_item = synthetic_data[i]
        # 从字符串格式的input中提取原始数值列表
        input_str = data_item['input']
        start_marker = "the given time series are as follow: "
        end_marker = "<\\s>"
        start_idx = input_str.find(start_marker) + len(start_marker)
        end_idx = input_str.find(end_marker)
        series_str = input_str[start_idx:end_idx]
        series_data = ast.literal_eval(series_str)
        output_label = data_item['output']

        # 计算实际的数据索引（从201开始）
        actual_index = start_index + i + 1

        # 生成图片文件名
        image_filename = f"pattern_samples/{actual_index}.png"

        try:
            plot_time_series(series_data, output_label, actual_index, image_filename)
            sample_count += 1

        except Exception as e:
            print(f"生成采样图片 {actual_index} 时出错: {e}")

    return sample_count


def generate_no_pattern_sample_images(no_pattern_data, start_index):
    """
    为No Temporal Pattern数据每隔20张生成采样图片

    Args:
        no_pattern_data: No Temporal Pattern数据列表
        start_index: 起始索引
    """
    create_no_pattern_samples_folder()

    sample_count = 0
    for i in range(0, len(no_pattern_data), 20):  # 每隔20张采样一次
        data_item = no_pattern_data[i]
        # 从字符串格式的input中提取原始数值列表
        input_str = data_item['input']
        start_marker = "the given time series are as follow: "
        end_marker = "<\\s>"
        start_idx = input_str.find(start_marker) + len(start_marker)
        end_idx = input_str.find(end_marker)
        series_str = input_str[start_idx:end_idx]

        # 清理字符串中的特殊标记
        series_str = series_str.replace('<|endoftext|', '').strip()

        try:
            series_data = ast.literal_eval(series_str)
        except (ValueError, SyntaxError) as e:
            print(f"解析时间序列数据时出错，索引 {i}: {e}")
            print(f"原始字符串: {series_str[:100]}...")
            continue
        output_label = data_item['output']

        # 计算实际的数据索引
        actual_index = start_index + i + 1


def generate_full_pattern_images(alpaca_data, dataset_dir, root_path):
    """
    为所有数据生成按模式分类的全量图片，并更新数据中的图片路径

    Args:
        alpaca_data: 原始的alpaca格式数据列表
        dataset_dir: 数据集目录
        root_path: 根路径

    Returns:
        更新了图片路径的数据列表
    """
    # 创建模式分类文件夹
    pattern_folders = create_pattern_folders(dataset_dir)

    # 按模式分类统计
    pattern_counters = {}
    for folder_name in pattern_folders.keys():
        pattern_counters[folder_name] = 0

    print("开始生成按模式分类的全量图片...")

    for index, data_item in tqdm(enumerate(alpaca_data), total=len(alpaca_data), desc="生成分类图片", unit="张"):
        # 新格式不包含时间序列数值，需要从CSV重新读取或生成模拟数据
        # 这里生成一个简单的模拟序列用于图片生成
        series_data = [0.1 * i + np.random.normal(0, 0.1) for i in range(100)]

        # 获取模式信息
        output_label = data_item['output']
        if isinstance(output_label, str):
            pattern_dict = extract_patterns_from_output(output_label)
            pattern = pattern_dict['Primary_Pattern']
        else:
            pattern = output_label.get('Primary_Pattern', 'No Temporal Pattern')

        # 获取对应的文件夹名称
        folder_name = get_pattern_folder_name(pattern)

        # 更新计数器
        pattern_counters[folder_name] += 1

        # 生成图片文件名（使用计数器作为编号）
        local_image_path = os.path.join(dataset_dir, folder_name, f"{pattern_counters[folder_name]}.png")
        full_image_path = root_path + f"{folder_name}/{pattern_counters[folder_name]}.png"

        try:
            # 生成图片时不显示模式标签
            plot_time_series(series_data, output_label, pattern_counters[folder_name], local_image_path,
                             show_pattern_label=False)

            # 更新数据中的图片路径（使用完整路径）
            alpaca_data[index]['images'] = [full_image_path]

        except Exception as e:
            print(f"生成分类图片 {local_image_path} 时出错: {e}")

    # 输出统计信息
    print("\n分类图片生成完成，各模式图片数量:")
    for folder_name, count in pattern_counters.items():
        if count > 0:
            print(f"  {folder_name}: {count} 张")

    return alpaca_data, pattern_counters


def generate_all_pattern_images(all_data, dataset_dir, root_path, selected_patterns=None):
    """
    为所有数据生成按模式分类的图片

    Args:
        all_data: 包含所有数据的列表（原始+合成+No Temporal Pattern）
        dataset_dir: 数据集目录
        root_path: 根路径
        selected_patterns: 指定要生成图片的模式类型列表，如果为None则生成所有模式
    """
    # 创建模式分类文件夹
    pattern_folders = create_pattern_folders(dataset_dir)

    # 按模式分类统计
    pattern_counters = {}
    for folder_name in pattern_folders.keys():
        pattern_counters[folder_name] = 0

    for index, data_item in tqdm(enumerate(all_data), total=len(all_data), desc="生成所有分类图片", unit="张"):
        # 新格式不包含时间序列数值，需要根据模式生成相应的模拟数据用于图片生成
        output_label = data_item['output']
        if isinstance(output_label, str):
            pattern_dict = extract_patterns_from_output(output_label)
            pattern = pattern_dict['Primary_Pattern']
        else:
            pattern = output_label.get('Primary_Pattern', 'No Temporal Pattern')

        # 如果指定了特定模式，只处理这些模式的数据
        if selected_patterns and pattern not in selected_patterns:
            continue

        # 根据模式类型生成相应的模拟时间序列数据，增加多样性
        series_length = np.random.randint(50, 150)  # 变长度
        x = np.arange(series_length)

        if 'Sudden Spike Outlier' in pattern:
            # 增强的突发异常值生成，增加周期性突变的丰富性
            base_level = np.random.uniform(-1, 1)
            noise_level = np.random.uniform(0.02, 0.08)  # 降低噪声

            # 40%概率生成有周期性背景的突发异常，60%概率生成纯随机背景的突发异常
            if np.random.random() < 0.4:
                # 在周期性背景上添加突发异常，增加丰富性
                period = np.random.randint(15, 30)
                amplitude = np.random.uniform(0.3, 0.8)
                phase = np.random.uniform(0, 2 * np.pi)
                x = np.arange(series_length)
                series_data = base_level + amplitude * np.sin(2 * np.pi * x / period + phase) + np.random.normal(0, noise_level, series_length)
            else:
                # 纯随机背景
                series_data = base_level + np.random.normal(0, noise_level, series_length)

            # 可能有多个尖峰
            num_spikes = np.random.randint(1, 4)
            for _ in range(num_spikes):
                # 根据时间序列长度动态调整spike位置范围
                min_pos = min(5, series_length // 6)
                max_pos = max(series_length - 5, series_length * 5 // 6)
                spike_pos = np.random.randint(min_pos, max_pos)
                spike_magnitude = np.random.choice([-1, 1]) * np.random.uniform(2, 6)
                spike_width = np.random.randint(1, 3)  # 尖峰宽度
                for i in range(spike_width):
                    if spike_pos + i < series_length:
                        series_data[spike_pos + i] += spike_magnitude * (1 - i / spike_width)

        elif 'Level Shift Outlier' in pattern:
            # 增强的水平偏移异常生成，增加周期性突变的丰富性
            base_level = np.random.uniform(-1, 1)
            noise_level = np.random.uniform(0.02, 0.08)  # 降低噪声

            # 40%概率生成有周期性背景的水平偏移异常，60%概率生成纯随机背景的水平偏移异常
            if np.random.random() < 0.4:
                # 在周期性背景上添加水平偏移异常，增加丰富性
                period = np.random.randint(15, 30)
                amplitude = np.random.uniform(0.3, 0.8)
                phase = np.random.uniform(0, 2 * np.pi)
                x = np.arange(series_length)
                series_data = base_level + amplitude * np.sin(2 * np.pi * x / period + phase) + np.random.normal(0, noise_level, series_length)
            else:
                # 纯随机背景
                series_data = base_level + np.random.normal(0, noise_level, series_length)

            # 可能有多个水平偏移
            num_shifts = np.random.randint(1, 3)
            # 根据时间序列长度动态调整shift位置范围
            min_pos = min(10, series_length // 4)
            max_pos = max(series_length - 10, series_length * 3 // 4)

            if max_pos > min_pos:
                shift_positions = sorted(np.random.choice(range(min_pos, max_pos), num_shifts, replace=False))
            else:
                # 如果序列太短，手动设置shift位置
                shift_positions = [series_length // 2]

            for shift_pos in shift_positions:
                shift_magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.8, 2.0)
                series_data[shift_pos:] += shift_magnitude

        elif 'Upward Trend' in pattern:
            # 多样化的上升趋势
            trend_slope = np.random.uniform(0.005, 0.05)  # 不同斜率
            base_level = np.random.uniform(-2, 2)  # 不同起始点
            noise_level = np.random.uniform(0.02, 0.08)  # 降低噪声水平

            # 40%概率生成周期性上涨趋势
            if np.random.random() < 0.4:
                # 周期性上涨：整体上升趋势 + 周期性波动
                period = np.random.randint(15, 35)
                cycle_amplitude = np.random.uniform(0.2, 0.6)  # 周期振幅相对较小
                phase = np.random.uniform(0, 2 * np.pi)
                series_data = (base_level + trend_slope * x +
                               cycle_amplitude * np.sin(2 * np.pi * x / period + phase) +
                               np.random.normal(0, noise_level, series_length))
            elif np.random.random() < 0.3:
                # 非线性上升趋势
                series_data = base_level + trend_slope * x + 0.001 * x ** 1.5 + np.random.normal(0, noise_level,
                                                                                                 series_length)
            else:
                # 线性上升趋势
                series_data = base_level + trend_slope * x + np.random.normal(0, noise_level, series_length)

        elif 'Downward Trend' in pattern:
            # 多样化的下降趋势
            trend_slope = np.random.uniform(-0.05, -0.005)
            base_level = np.random.uniform(-2, 2)
            noise_level = np.random.uniform(0.02, 0.08)  # 降低噪声水平

            # 40%概率生成周期性下降趋势
            if np.random.random() < 0.4:
                # 周期性下降：整体下降趋势 + 周期性波动
                period = np.random.randint(15, 35)
                cycle_amplitude = np.random.uniform(0.2, 0.6)  # 周期振幅相对较小
                phase = np.random.uniform(0, 2 * np.pi)
                series_data = (base_level + trend_slope * x +
                               cycle_amplitude * np.sin(2 * np.pi * x / period + phase) +
                               np.random.normal(0, noise_level, series_length))
            elif np.random.random() < 0.3:
                # 非线性下降趋势
                series_data = base_level + trend_slope * x - 0.001 * x ** 1.5 + np.random.normal(0, noise_level,
                                                                                                 series_length)
            else:
                # 线性下降趋势
                series_data = base_level + trend_slope * x + np.random.normal(0, noise_level, series_length)

        elif 'Fixed Seasonality' in pattern:
            # 多样化的固定季节性
            base_level = np.random.uniform(-1, 1)
            # 确保至少有2个完整周期：period * 2 <= series_length
            max_period = series_length // 2
            period = np.random.randint(12, min(30, max_period))  # 不同周期
            amplitude = np.random.uniform(0.5, 2.0)  # 不同幅度
            phase = np.random.uniform(0, 2 * np.pi)  # 不同相位
            noise_level = np.random.uniform(0.01, 0.06)  # 降低噪声

            # 可能是正弦波或余弦波的组合
            if np.random.random() < 0.5:
                series_data = base_level + amplitude * np.sin(2 * np.pi * x / period + phase)
            else:
                series_data = base_level + amplitude * np.cos(2 * np.pi * x / period + phase)

            # 可能添加谐波
            if np.random.random() < 0.3:
                harmonic_amplitude = amplitude * np.random.uniform(0.2, 0.5)
                series_data += harmonic_amplitude * np.sin(4 * np.pi * x / period + phase)

            series_data += np.random.normal(0, noise_level, series_length)

        elif 'Shifting Seasonality' in pattern:
            # 多样化的变化季节性
            base_level = np.random.uniform(-1, 1)
            # 确保至少有2个完整周期：period * 2 <= series_length
            max_period = series_length // 2
            period = np.random.randint(12, min(30, max_period))
            base_amplitude = np.random.uniform(0.3, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            noise_level = np.random.uniform(0.01, 0.06)  # 降低噪声

            # 随机选择幅度变化方向：增大或减小
            amplitude_direction = np.random.choice(['increasing', 'decreasing'])

            if amplitude_direction == 'increasing':
                # 幅度从小逐渐变大
                amplitude_change = np.random.uniform(0.5, 2.0)
                if np.random.random() < 0.5:
                    # 线性变化
                    amplitude = base_amplitude + amplitude_change * (x / series_length)
                else:
                    # 非线性变化
                    amplitude = base_amplitude + amplitude_change * ((x / series_length) ** 1.5)
            else:
                # 幅度从大逐渐变小
                amplitude_change = np.random.uniform(0.5, 2.0)
                if np.random.random() < 0.5:
                    # 线性变化
                    amplitude = base_level + amplitude_change - amplitude_change * (x / series_length)
                else:
                    # 非线性变化
                    amplitude = base_amplitude + amplitude_change - amplitude_change * ((x / series_length) ** 1.5)

            # 确保幅度不为负数
            amplitude = np.maximum(amplitude, 0.1)

            series_data = base_level + amplitude * np.sin(2 * np.pi * x / period + phase) + np.random.normal(0,
                                                                                                             noise_level,
                                                                                                             series_length)
        elif 'Obvious Volatility' in pattern:
            # 生成明显波动性数据
            base_level = np.random.uniform(-1, 1)
            noise_level = np.random.uniform(0.2, 0.5)  # 增加噪声水平以体现波动性
            
            # 60%概率生成高波动随机数据，40%概率生成随机游走
            if np.random.random() < 0.6:
                # 高波动随机数据：围绕基准水平的大幅随机变化
                series_data = base_level + np.random.normal(0, noise_level, series_length)
            else:
                # 随机游走：较大的步长变化
                series_data = np.zeros(series_length)
                series_data[0] = base_level
                step_size = noise_level * 0.4  # 较大的步长
                for i in range(1, series_length):
                    series_data[i] = series_data[i - 1] + np.random.normal(0, step_size)
        
        else:  # No Temporal Pattern
            # 生成完全平坦、无趋势、无周期、无规则的数据
            base_level = np.random.uniform(0, 0.5)  # 缩小基准水平范围
            
            # 大幅增加数据点数量来减少视觉上的波动
            series_length = np.random.randint(1200, 1400)  # 增加到200-400个数据点
            x = np.arange(series_length)
            
            # 随机选择生成类型：70%完全平直线，20%极微小随机噪声，10%超微小不规则变化
            pattern_type = np.random.choice(['flat_line', 'minimal_noise', 'micro_irregular'],
                                          p=[0.7, 0.2, 0.1])
            
            if pattern_type == 'flat_line':
                # 完全平直线：几乎没有任何变化
                noise_level = np.random.uniform(0, 0.0005)  # 极极极低噪声
                series_data = np.full(series_length, base_level) + np.random.normal(0, noise_level, series_length)
                
            elif pattern_type == 'minimal_noise':
                # 极微小随机噪声：完全无规则，无任何趋势或周期
                noise_level = np.random.uniform(0, 0.003)  # 极低噪声水平
                series_data = np.full(series_length, base_level) + np.random.normal(0, noise_level, series_length)
                
            else:  # micro_irregular
                # 超微小不规则变化：完全随机，无任何模式
                noise_level = np.random.uniform(0, 0.005)  # 仍然很低的噪声
                # 生成完全随机的微小变化，确保无任何趋势或周期性
                random_changes = np.random.normal(0, noise_level, series_length)
                # 去除任何可能的累积趋势
                random_changes = random_changes - np.linspace(random_changes[0], random_changes[-1], series_length)
                series_data = np.full(series_length, base_level) + random_changes

        # 获取模式信息
        output_label = data_item['output']
        if isinstance(output_label, str):
            pattern_dict = extract_patterns_from_output(output_label)
            pattern = pattern_dict['Primary_Pattern']
        else:
            pattern = output_label.get('Primary_Pattern', 'No Temporal Pattern')

        # 获取对应的文件夹名称
        folder_name = get_pattern_folder_name(pattern)

        # 更新计数器
        pattern_counters[folder_name] += 1

        # 生成图片文件名（使用计数器作为编号）
        local_image_path = os.path.join(dataset_dir, folder_name, f"{pattern_counters[folder_name]}.png")
        full_image_path = root_path + f"{folder_name}/{pattern_counters[folder_name]}.png"

        try:
            # 生成图片时不显示模式标签
            plot_time_series(series_data, output_label, pattern_counters[folder_name], local_image_path,
                             show_pattern_label=False)

            # 更新数据中的图片路径（使用完整路径）
            all_data[index]['images'] = [full_image_path]


        except Exception as e:
            print(f"生成分类图片 {local_image_path} 时出错: {e}")

    # 输出统计信息
    print("\n所有数据分类图片生成完成，各模式图片数量:")
    for folder_name, count in pattern_counters.items():
        if count > 0:
            print(f"  {folder_name}: {count} 张")

    return all_data, pattern_counters