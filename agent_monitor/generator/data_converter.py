"""
数据格式转换模块
包含标签转换、模式提取等数据格式处理功能
"""

import csv
import ast


def label_to_alpaca_format(label):
    """
    将Label转换为Alpaca格式的输出
    根据用户要求，输出格式为："Primary pattern: XXX; sub-pattern: YYY.<\\s>"
    """
    # 根据CSV数据分析，Label列包含的值如：upward trend, downward trend 等
    # 需要映射到层次化的模式分类

    label_mapping = {
        'upward trend': 'Upward Trend<\\s>',
        'downward trend': 'Downward Trend<\\s>',
        'sudden spike outlier': 'Sudden Spike Outlier<\\s>',
        'level shift outlier': 'Level Shift Outlier<\\s>',
        'fixed seasonal': 'Fixed Seasonality<\\s>',
        'shifting seasonal': 'Shifting Seasonality<\\s>',
        'increased volatility': 'Obvious Volatility<\\s>',
        'decreased volatility': 'Obvious Volatility<\\s>',
        'obvious volatility': 'Obvious Volatility<\\s>',
        'no temporal pattern': 'No Temporal Pattern<\\s>'
    }

    return label_mapping.get(label.lower(), 'No Temporal Pattern<\\s>')


def extract_patterns_from_output(output_text):
    """
    从输出字符串中提取Primary_Pattern和Sub_Pattern
    输入格式: "Primary pattern: Trend; sub-pattern: Upward Trend.<\\s>"
    """
    try:
        # 移除<\\s>标记
        label = output_text.replace('<\\s>', '').strip()
        return {
            'Primary_Pattern': label,
            'Sub_Pattern': label
        }
    except:
        pass

    # 如果解析失败，返回默认值
    return {
        'Primary_Pattern': 'No Temporal Pattern',
        'Sub_Pattern': 'No Temporal Pattern'
    }


def get_pattern_folder_name(pattern):
    """
    根据模式名称获取对应的文件夹名称
    """
    pattern_mapping = {
        'No Temporal Pattern': 'no_temporal_pattern',
        'Sudden Spike Outlier': 'pattern_sudden_spike_outlier',
        'Level Shift Outlier': 'pattern_level_shift_outlier',
        'Upward Trend': 'pattern_upward_trend',
        'Downward Trend': 'pattern_downward_trend',
        'Fixed Seasonality': 'pattern_fixed_seasonality',
        'Shifting Seasonality': 'pattern_shifting_seasonality',
        'Obvious Volatility': 'pattern_obvious_volatility'
    }
    return pattern_mapping.get(pattern, 'no_temporal_pattern')


def create_alpaca_format(instruction, generate_images=True):
    """
    读取TSQA.csv文件并生成Alpaca格式的JSON数据，同时可选择生成可视化图片

    Args:
        instruction: 指令文本
        generate_images: 是否生成可视化图片
    """
    alpaca_data = []
    global_position = 0  # 全局位置计数器，用于计算range

    # 不再生成pattern_200目录的图片

    try:
        with open('TSQA.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)

            for index, row in enumerate(csv_reader):
                # 获取Series列（作为input）和Label列（作为output）
                series_data = row['Series']
                label = row['Label']

                # 将字符串形式的列表转换为实际的列表
                try:
                    # 使用ast.literal_eval安全地解析字符串形式的列表
                    series_list = ast.literal_eval(series_data)
                except (ValueError, SyntaxError) as e:
                    print(f"解析Series数据时出错: {e}")
                    continue

                # 计算当前时间序列的长度
                series_length = len(series_list)
                
                # 计算range属性：[start, end]
                range_start = global_position + 1  # 从1开始计数
                range_end = global_position + series_length
                range_info = [range_start, range_end]
                
                # 更新全局位置计数器
                global_position += series_length

                # 转换Label为Alpaca格式
                output_format = label_to_alpaca_format(label)

                # 新的input格式，不包含时间序列数值
                input_str = """Focus on the overall shape, continuity, and repetition of the line.
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
Output strictly as: {"Label": "<one_of_the_above_labels>"}."""

                # 创建Alpaca格式的条目
                alpaca_entry = {
                    "instruction": instruction,
                    "input": input_str,
                    "output": output_format,
                    "images": [],  # 图片路径数组将在后续填充
                    "range": range_info  # 添加range属性：[start, end]
                }

                alpaca_data.append(alpaca_entry)

                # 不再生成pattern_200目录的图片

        return alpaca_data

    except FileNotFoundError:
        print("错误：找不到TSQA.csv文件")
        return []
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return []