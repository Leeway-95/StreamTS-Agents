import os
import sys
import pandas as pd
import glob
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *

# RAGAS imports for similarity calculation
try:
    from ragas.metrics import answer_similarity
    from ragas import evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    # print("Warning: RAGAS not available. Falling back to SequenceMatcher for similarity calculation.")
    RAGAS_AVAILABLE = False

FONT_SIZE = 8

def safe_eval_list(data_str):
    """Safely parse string to list"""
    if pd.isna(data_str) or data_str == '[]' or data_str == '':
        return []
    
    # Convert to string and clean
    data_str = str(data_str).strip()
    if not data_str or data_str == '[]':
        return []
    
    try:
        # Try using ast.literal_eval directly
        result = ast.literal_eval(data_str)
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        try:
            # Try using json.loads
            result = json.loads(data_str)
            return result if isinstance(result, list) else [result]
        except (json.JSONDecodeError, ValueError):
            try:
                # Try to fix common JSON format issues
                # Replace single quotes with double quotes
                fixed_str = data_str.replace("'", '"')
                result = json.loads(fixed_str)
                return result if isinstance(result, list) else [result]
            except (json.JSONDecodeError, ValueError):
                # If still fails, try to handle as single string
                if data_str.startswith('[') and data_str.endswith(']'):
                    # If it looks like a list but parsing fails, return empty list
                    print(f"Warning: Failed to parse list-like string: {data_str[:100]}...")
                    return []
                else:
                    # If not in list format, return as single element
                    return [data_str]


def calculate_mae(pred_series, truth_series):
    """Calculate Mean Absolute Error (MAE)"""
    if not pred_series or not truth_series:
        return 0.0
    
    pred_array = np.array(pred_series)
    truth_array = np.array(truth_series)
    
    if len(pred_array) != len(truth_array):
        return 0.0
    
    return np.mean(np.abs(pred_array - truth_array))


def calculate_ragas_similarity(pred_labels, truth_labels):
    """使用RAGAS计算文本相似度"""
    # 检查输入是否为空或无效
    if not pred_labels or not truth_labels:
        return 0.0
    
    # 确保输入是列表
    if not isinstance(pred_labels, list):
        pred_labels = [pred_labels] if pred_labels else []
    if not isinstance(truth_labels, list):
        truth_labels = [truth_labels] if truth_labels else []
    
    # 再次检查是否为空
    if len(pred_labels) == 0 or len(truth_labels) == 0:
        return 0.0
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 将列表转换为有意义的字符串
        pred_str = ' '.join(str(label).strip() for label in pred_labels if label)
        truth_str = ' '.join(str(label).strip() for label in truth_labels if label)
        
        # 如果文本为空，返回0
        if not pred_str.strip() or not truth_str.strip():
            return 0.0
        
        # 使用sentence-transformers计算语义相似度
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 编码文本
        pred_embedding = model.encode([pred_str])
        truth_embedding = model.encode([truth_str])
        
        # 计算余弦相似度
        similarity = cosine_similarity(pred_embedding, truth_embedding)[0][0]
        
        return float(similarity)
    
    except ImportError:
        print("Warning: sentence-transformers not available. Using fallback method.")
        return calculate_text_similarity_fallback(pred_labels, truth_labels)
    except Exception as e:
        print(f"Error using semantic similarity calculation: {str(e)}")
        # 如果语义相似度计算失败，回退到SequenceMatcher
        return calculate_text_similarity_fallback(pred_labels, truth_labels)


def calculate_text_similarity_fallback(pred_labels, truth_labels):
    """使用SequenceMatcher作为后备的文本相似度计算"""
    # 检查输入是否为空或无效
    if not pred_labels or not truth_labels:
        return 0.0
    
    # 确保输入是列表
    if not isinstance(pred_labels, list):
        pred_labels = [pred_labels] if pred_labels else []
    if not isinstance(truth_labels, list):
        truth_labels = [truth_labels] if truth_labels else []
    
    # 再次检查是否为空
    if len(pred_labels) == 0 or len(truth_labels) == 0:
        return 0.0
    
    # 将列表转换为有意义的字符串进行比较
    pred_str = ' '.join(str(label).strip() for label in pred_labels if label)
    truth_str = ' '.join(str(label).strip() for label in truth_labels if label)
    
    # 如果文本为空，返回0
    if not pred_str.strip() or not truth_str.strip():
        return 0.0
    
    from difflib import SequenceMatcher
    # 使用 SequenceMatcher 计算相似度
    similarity = SequenceMatcher(None, pred_str, truth_str).ratio()
    return similarity


def calculate_text_similarity(pred_labels, truth_labels):
    """计算文本相似度（用于 REASONING 任务）- 优先使用RAGAS"""
    if not pred_labels or not truth_labels:
        return 0.0
    
    if RAGAS_AVAILABLE:
        return calculate_ragas_similarity(pred_labels, truth_labels)
    else:
        return calculate_text_similarity_fallback(pred_labels, truth_labels)


def calculate_position_accuracy(pred_labels, truth_labels):
    """计算位置匹配准确率（用于 UNDERSTANDING 任务）"""
    if not pred_labels or not truth_labels:
        return 0.0
    
    # 确保两个列表长度相同
    min_len = min(len(pred_labels), len(truth_labels))
    if min_len == 0:
        return 0.0
    
    # 计算位置匹配的准确率
    matches = sum(1 for i in range(min_len) if pred_labels[i] == truth_labels[i])
    return matches / len(truth_labels)


def calculate_event_metrics(pred_labels, truth_labels, impact_scores=None):
    """计算事件预测的 F1 和 AUC 指标（用于 FORECASTING_EVENT 任务）"""
    if not pred_labels or not truth_labels:
        return {"f1": 0.0, "auc": 0.0, "accuracy": 0.0}
    
    try:
        def label_to_binary(label):
            """将标签转换为二进制值，支持Weather数据集的'rained'/'not rained'格式和LLM返回的JSON格式"""
            if isinstance(label, bool):
                return int(label)
            elif isinstance(label, str):
                # 首先尝试解析JSON格式的LLM响应
                try:
                    import json
                    if label.strip().startswith('{') and label.strip().endswith('}'):
                        parsed = json.loads(label)
                        if 'Pred_Labels' in parsed:
                            label = parsed['Pred_Labels']
                except (json.JSONDecodeError, ValueError):
                    pass
                
                label_lower = str(label).lower().strip()
                
                # 支持Healthcare数据集的格式 - 优先匹配exceed相关
                if 'exceed' in label_lower:
                    if 'not exceed' in label_lower or 'did not exceed' in label_lower or 'did not exceed the average' in label_lower:
                        return 0  # 没有超过 = 负类
                    else:
                        return 1  # 超过 = 正类
                # 支持Weather数据集的格式
                elif 'rained' in label_lower:
                    return 1 if label_lower == 'rained' else 0
                elif 'rain' in label_lower:
                    return 0 if 'not' in label_lower else 1
                # 支持Finance数据集的格式
                elif 'up' in label_lower or 'rise' in label_lower or 'increase' in label_lower:
                    return 1
                elif 'down' in label_lower or 'fall' in label_lower or 'decrease' in label_lower:
                    return 0
                # 支持Healthcare数据集的其他格式
                elif 'high' in label_lower or 'mortality' in label_lower:
                    return 1 if 'high' in label_lower else 0
                elif 'positive' in label_lower or 'negative' in label_lower:
                    return 1 if 'positive' in label_lower else 0
                # 支持通用格式
                elif label_lower in ['true', '1', 'yes', 'positive', 'event', 'occurred']:
                    return 1
                elif label_lower in ['false', '0', 'no', 'negative', 'no event', 'not occurred']:
                    return 0
                else:
                    return 0  # 默认为负类
            else:
                return int(bool(label))
        
        # 处理预测标签
        if isinstance(pred_labels, (list, tuple)):
            pred_binary = [label_to_binary(label) for label in pred_labels]
        else:
            pred_binary = [label_to_binary(pred_labels)]
        
        # 处理真实标签
        if isinstance(truth_labels, (list, tuple)):
            truth_binary = [label_to_binary(label) for label in truth_labels]
        else:
            truth_binary = [label_to_binary(truth_labels)]
        
        # 确保长度一致
        min_len = min(len(pred_binary), len(truth_binary))
        if min_len == 0:
            return {"f1": 0.0, "auc": 0.0, "accuracy": 0.0}
        
        pred_binary = pred_binary[:min_len]
        truth_binary = truth_binary[:min_len]
        
        # 计算 F1 分数
        f1 = f1_score(truth_binary, pred_binary, average='binary', zero_division=0)
        
        # 计算准确率
        accuracy = accuracy_score(truth_binary, pred_binary)
        
        # 计算 AUC（如果有 impact_scores）
        auc = 0.0
        if impact_scores and len(impact_scores) >= min_len:
            try:
                # 使用 impact_scores 作为概率分数计算 AUC
                scores = impact_scores[:min_len]
                # 确保分数在 [0, 1] 范围内
                scores = [max(0, min(1, float(score))) for score in scores]
                if len(set(truth_binary)) > 1:  # 确保有正负样本
                    auc = roc_auc_score(truth_binary, scores)
            except Exception as e:
                print(f"Warning: Failed to calculate AUC: {e}")
                auc = 0.0
        else:
            # 如果没有 impact_scores，使用预测概率
            try:
                if len(set(truth_binary)) > 1:  # 确保有正负样本
                    auc = roc_auc_score(truth_binary, pred_binary)
            except Exception as e:
                print(f"Warning: Failed to calculate AUC with predictions: {e}")
                auc = 0.0
        
        return {"f1": float(f1), "auc": float(auc), "accuracy": float(accuracy)}
        
    except Exception as e:
        print(f"Error calculating event metrics: {e}")
        return {"f1": 0.0, "auc": 0.0, "accuracy": 0.0}


def create_forecasting_num_plot(pred_series, truth_series, original_series, row_index, save_path, dataset, method):
    """为 FORECASTING_NUM 任务创建时间序列图"""
    try:
        plt.figure(figsize=(12, 6))
        
        # 创建时间轴
        hist_len = len(original_series) if original_series else 0
        pred_len = len(pred_series) if pred_series else 0
        
        # 原始序列不再绘制（统一所有方法均不绘制 Original Series）
        # if original_series:
        #     x_orig = range(hist_len)
        #     plt.plot(x_orig, original_series, 'b-', label='Original Series', linewidth=2)
        
        # 绘制预测序列（红色）
        if pred_series:
            x_pred = range(hist_len, hist_len + pred_len)
            plt.plot(x_pred, pred_series, 'r-', label='Predicted Series', linewidth=2)
        
        # 绘制真实序列（黄色）
        if truth_series:
            x_truth = range(hist_len, hist_len + len(truth_series))
            plt.plot(x_truth, truth_series, 'y-', label='Ground Truth Series', linewidth=2)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.title(f'Row {row_index+1} - Time Series FORECASTING_NUM - Dataset: {dataset}, Method: {method}')

        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error creating plot for row {row_index}: {str(e)}")
        plt.close()
        return False


def get_original_series_from_stream(dataset_name, row_index):
    """从 stream_summary.csv 获取原始时间序列数据"""
    try:
        # 使用绝对路径构建stream_summary.csv路径
        datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        stream_path = os.path.join(datasets_path, dataset_name, f"stream-{dataset_name}", "stream_summary.csv")
        if os.path.exists(stream_path):
            stream_df = pd.read_csv(stream_path)
            if row_index < len(stream_df) and 'Series' in stream_df.columns:
                series_str = stream_df.iloc[row_index]['Series']
                return safe_eval_list(series_str)
    except Exception as e:
        print(f"Error reading stream data for {dataset_name}, row {row_index}: {str(e)}")
    
    return []


def draw_understanding_plots(predict_compare_path, stream_summary_path, output_dir):
    """
    绘制理解任务的可视化图表

    Args:
        predict_compare_path: predict_compare.csv文件路径
        stream_summary_path: stream_summary.csv文件路径
        output_dir: 输出目录
    """
    # 读取数据
    df = pd.read_csv(predict_compare_path)
    stream_df = pd.read_csv(stream_summary_path)

    os.makedirs(output_dir, exist_ok=True)

    # 过滤出 UNDERSTANDING 任务的数据
    understanding_df = df[df['Task'] == 'UNDERSTANDING']

    for index, row in understanding_df.iterrows():
        # 对于Understanding任务，所有方法都使用同一个数据集的stream数据
        # 使用第一行stream数据（索引0），因为所有方法处理的是同一个时间序列
        if len(stream_df) == 0:
            continue
        stream_row = stream_df.iloc[0]  # 所有方法使用同一行stream数据

        # Parse data series - 从stream_summary.csv读取
        hist_series = json.loads(stream_row['Series'])
        hist_position = ast.literal_eval(stream_row['Positions'])

        # Parse labels - 从predict_compare.csv读取
        true_labels = ast.literal_eval(row['Pred_Labels_Truth']) if pd.notna(row['Pred_Labels_Truth']) else []
        pred_labels = ast.literal_eval(row['Pred_Labels']) if pd.notna(row['Pred_Labels']) else []

        # Create chart for UNDERSTANDING task (without Representative Subsequence)
        plt.figure(figsize=(28, 7))

        # Plot historical data
        plt.plot(range(len(hist_series)), hist_series,
                 color='#1f77b4', linewidth=1.5, label='Streaming Time Series')

        # Calculate label positions and vertical offsets
        y_min, y_max = plt.ylim()
        offset = 0.05 * (y_max - y_min)
        positions = []
        for (start, end) in hist_position:
            positions.append((start + end) / 2)
        positions.sort()

        # Calculate label grouping levels to avoid overlap
        group_level = np.zeros(len(positions), dtype=int)
        if positions:
            group_level[0] = 0
            for i in range(1, len(positions)):
                if positions[i] - positions[i - 1] < (y_max - y_min) / 20:
                    group_level[i] = group_level[i - 1] + 1
                else:
                    group_level[i] = group_level[i - 1] if group_level[i - 1] > 0 else 0
        max_level = max(group_level) if positions else 0

        # Draw labels for UNDERSTANDING task
        for idx, ((start, end), true_label, pred_label) in enumerate(zip(hist_position, true_labels, pred_labels)):
            mid = (start + end) / 2
            level = group_level[positions.index(mid)]
            vertical_offset = (level * offset * 1.5)

            # Alternate label positions above and below the series
            if idx % 2 == 0:
                true_y = y_min - offset - vertical_offset
                pred_y = y_min - 2.0 * offset - vertical_offset
                arrow_start_y = y_min - 0.5 * offset
            else:
                true_y = y_max + offset + vertical_offset
                pred_y = y_max + 2.0 * offset + vertical_offset
                arrow_start_y = y_max + 0.5 * offset

            # Display true and predicted labels
            plt.text(mid, true_y, f'True: {true_label}', ha='center', fontsize=FONT_SIZE,
                     color='black', backgroundcolor='white', alpha=0.8)

            # Color predicted label based on correctness
            color = 'red' if true_label != pred_label else 'green'
            plt.text(mid, pred_y, f'Pred: {pred_label}', ha='center', fontsize=FONT_SIZE,
                     color=color, backgroundcolor='white', alpha=0.8)

            # Draw arrow pointing to the labeled region
            # Find the corresponding y-value in the series for the arrow target
            if mid >= 0 and mid < len(hist_series):
                # Point is in historical series
                series_idx = int(mid)
                series_y = hist_series[min(max(0, series_idx), len(hist_series) - 1)]
            else:
                # Default to middle of y-range if position is unclear
                series_y = (y_min + y_max) / 2

            plt.annotate('', xy=(mid, series_y), xytext=(mid, arrow_start_y),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.7))

            # Add vertical lines to mark label boundaries
            plt.axvline(x=start, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            plt.axvline(x=end, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Adjust plot limits to accommodate labels
        plt.ylim(y_min - (max_level + 3.0) * offset, y_max + (max_level + 3.0) * offset)

        # Set title and labels
        plt.title(f'Row {row["Index"]+1} - Time Series UNDERSTANDING - Dataset: {row["Dataset"]}, Method: {row["Method"]}',
                  fontsize=FONT_SIZE + 2)
        plt.xlabel('Time', fontsize=FONT_SIZE)
        plt.ylabel('Value', fontsize=FONT_SIZE)
        plt.legend(fontsize=FONT_SIZE)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot
        output_filename = f'row_{row["Index"]}.png'
        plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close()

def analyze_predict_compare_results(custom_files=None):
    """
    遍历datasets目录下所有predict_compare.csv文件，按照Dataset，Method，预测长度，
    平均Pred_Labels_Accuracy，平均Pred_Series_MAE来统计并写入到logs目录下的metrics_exp_res.csv文件中
    
    Args:
        custom_files: 可选的自定义文件列表，用于测试
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(LOG_EXP_METRICS_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 存储所有数据
    all_data = []

    # 如果提供了自定义文件列表，使用它；否则查找所有predict_compare.csv文件
    if custom_files:
        predict_files = custom_files
    else:
        # 获取所有数据集路径 - 使用绝对路径
        datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        # 查找所有predict_compare.csv文件
        predict_files = glob.glob(os.path.join(datasets_path, "*/predict-*/predict_compare.csv"))

    if not predict_files:
        print("No predict_compare.csv files found")
        return

    idx = 1
    file_progress = tqdm(predict_files, desc="Processing files", leave=True)
    for file_path in file_progress:
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
            # print(f"Processing file: {file_path}")

            # 检查是否存在必要的列
            required_columns = ['Task', 'Dataset', 'Method']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns} in {file_path}")
                continue
            
            # 如果没有Task列，根据Method推断Task类型
            if 'Task' not in df.columns:
                def infer_task_from_method(method):
                    if method in BASELINE_UNDERSTANDING:
                        return "UNDERSTANDING"
                    elif method in BASELINE_FORECASTING_NUM:
                        return "FORECASTING_NUM"
                    elif method in BASELINE_FORECASTING_EVENT:
                        return "FORECASTING_EVENT"
                    elif method in BASELINE_REASONING:
                        return "REASONING"
                    else:
                        return "Unknown"
                
                df['Task'] = df['Method'].apply(infer_task_from_method)

            # 获取所有唯一的方法和数据集名称
            unique_methods = df['Method'].unique()
            unique_datasets = df['Dataset'].unique()

            # 按照预测长度分组处理数据
            unique_lengths = df['Pred_Series_Len'].unique()

            # 对每个预测长度计算指标
            for pred_len in unique_lengths:
                # 筛选当前预测长度的数据
                length_df = df[df['Pred_Series_Len'] == pred_len]

                # 按数据集和方法分组处理
                for dataset in unique_datasets:
                    dataset_df = length_df[length_df['Dataset'] == dataset]
                    
                    if dataset_df.empty:
                        continue
                    
                    # 获取该数据集中的所有方法
                    dataset_methods = dataset_df['Method'].unique()
                    
                    # 如果有Task列，按Task和Method同时分组
                    if 'Task' in dataset_df.columns:
                        # 按Task和Method同时分组处理
                        for (task, method), method_df in dataset_df.groupby(['Task', 'Method']):
                            if method_df.empty:
                                continue
                            
                            # 标记是否需要更新原始CSV文件
                            csv_updated = False
                            
                            # 为每一行数据进行绘图和指标计算
                            for row_idx, row in method_df.iterrows():
                                # 解析数据
                                pred_series = safe_eval_list(row.get('Pred_Series', '[]'))
                                truth_series = safe_eval_list(row.get('Pred_Series_Truth', '[]'))
                                pred_labels = safe_eval_list(row.get('Pred_Labels', '[]'))
                                truth_labels = safe_eval_list(row.get('Pred_Labels_Truth', '[]'))
                                
                                # 获取原始序列数据（用于绘图）
                                original_series = get_original_series_from_stream(dataset, row_idx - 1)
                                draw_dir = ''
                                if task == 'FORECASTING_NUM' or task == 'UNDERSTANDING':
                                    # 构建保存路径
                                    predict_dir = os.path.dirname(file_path)
                                    draw_dir = os.path.join(predict_dir, 'draw')
                                    plot_path = os.path.join(draw_dir, f'{dataset}_{method}_row_{row_idx}.png')

                                # 为 FORECASTING_NUM 任务生成绘图
                                if task == 'FORECASTING_NUM' and pred_series and truth_series and OUTPUT_PREDICT_IMAGE:
                                    
                                    # 创建绘图
                                    create_forecasting_num_plot(pred_series, truth_series, original_series, row_idx, plot_path, dataset, method)
                                    # print(f"Created plot for {task} task, row {row_idx}: {plot_path}")

                                if (task == 'UNDERSTANDING' or task == 'FORECASTING_EVENT') and pred_labels and truth_labels:
                                    file_dir = os.path.dirname(file_path)
                                    dataset_dir = os.path.dirname(file_dir)
                                    dataset_name = os.path.basename(dataset_dir)
                                    # 使用绝对路径构建stream_summary.csv路径
                                    datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
                                    stream_summary_path = os.path.join(datasets_path, dataset_name, f"stream-{dataset_name}", "stream_summary.csv")
                                    
                                    if os.path.exists(stream_summary_path) and OUTPUT_PREDICT_IMAGE:
                                        draw_understanding_plots(file_path, stream_summary_path, draw_dir)
                                    # print(f"Created plot for {task} task, row {row_idx}: {plot_path}")

                                if task == 'REASONING' and pred_labels and truth_labels and OUTPUT_PREDICT_IMAGE:
                                    # 构建保存路径
                                    predict_dir = os.path.dirname(file_path)
                                    draw_dir = os.path.join(predict_dir, 'draw')
                                    plot_path = os.path.join(draw_dir, f'{dataset}_{method}_row_{row_idx}.png')
                                
                                # 计算扩展指标并更新原始DataFrame
                                if task == 'FORECASTING_NUM':
                                    # 重新计算 MAE
                                    if pred_series and truth_series:
                                        calculated_mae = calculate_mae(pred_series, truth_series)
                                        df.loc[row_idx, 'Pred_Series_MAE'] = calculated_mae
                                        method_df.loc[row_idx, 'Pred_Series_MAE'] = calculated_mae
                                        csv_updated = True
                                
                                elif task == 'REASONING':
                                    # 计算文本相似度
                                    if pred_labels and truth_labels:
                                        text_similarity = calculate_text_similarity(pred_labels, truth_labels)
                                        df.loc[row_idx, 'Pred_Labels_Accuracy'] = text_similarity
                                        method_df.loc[row_idx, 'Pred_Labels_Accuracy'] = text_similarity
                                        csv_updated = True
                                
                                elif task == 'UNDERSTANDING':
                                    # 计算位置匹配准确率
                                    if pred_labels and truth_labels:
                                        position_accuracy = calculate_position_accuracy(pred_labels, truth_labels)
                                        df.loc[row_idx, 'Pred_Labels_Accuracy'] = position_accuracy
                                        method_df.loc[row_idx, 'Pred_Labels_Accuracy'] = position_accuracy
                                        csv_updated = True
                                
                                elif task == 'FORECASTING_EVENT':
                                    # 计算事件预测指标 (F1, AUC, Accuracy) - 不处理Pred_Series
                                    if pred_labels and truth_labels:
                                        # 尝试获取 Impact_Scores
                                        impact_scores = safe_eval_list(row.get('Impact_Scores', '[]'))
                                        
                                        # 计算事件预测指标
                                        event_metrics = calculate_event_metrics(pred_labels, truth_labels, impact_scores)
                                        
                                        # 更新 DataFrame - 不更新Pred_Series_MAE
                                        df.loc[row_idx, 'Pred_Labels_Accuracy'] = event_metrics['accuracy']
                                        method_df.loc[row_idx, 'Pred_Labels_Accuracy'] = event_metrics['accuracy']
                                        
                                        # 添加 F1Score 和 AUC 列（如果不存在）
                                        if 'Pred_Labels_F1Score' not in df.columns:
                                            df['Pred_Labels_F1Score'] = 0.0
                                        if 'Pred_Labels_AUC' not in df.columns:
                                            df['Pred_Labels_AUC'] = 0.0
                                        
                                        df.loc[row_idx, 'Pred_Labels_F1Score'] = event_metrics['f1']
                                        method_df.loc[row_idx, 'Pred_Labels_F1Score'] = event_metrics['f1']
                                        df.loc[row_idx, 'Pred_Labels_AUC'] = event_metrics['auc']
                                        method_df.loc[row_idx, 'Pred_Labels_AUC'] = event_metrics['auc']
                                        
                                        # 确保FORECASTING_EVENT任务的Pred_Series_MAE为0
                                        df.loc[row_idx, 'Pred_Series_MAE'] = 0.0
                                        method_df.loc[row_idx, 'Pred_Series_MAE'] = 0.0
                                        
                                        csv_updated = True
                            
                            # 如果更新了数据，保存回原始CSV文件
                            if csv_updated:
                                df.to_csv(file_path, index=False)
                                # print(f"Updated CSV file: {file_path}")
                            
                            # 计算平均指标
                            method_df = method_df.dropna(subset=['Pred_Series_Len', 'Pred_Series_Truth_Len'])
                            same_length_df = method_df[method_df['Pred_Series_Len'] == method_df['Pred_Series_Truth_Len']]

                            method_avg_accuracy = same_length_df[
                                'Pred_Labels_Accuracy'].mean() if 'Pred_Labels_Accuracy' in same_length_df.columns and not same_length_df.empty else 0
                            method_avg_mae = same_length_df[
                                'Pred_Series_MAE'].mean() if 'Pred_Series_MAE' in same_length_df.columns and not same_length_df.empty else 0
                            
                            # 计算F1和AUC的平均值（仅对FORECASTING_EVENT任务）
                            method_avg_f1 = 0.0
                            method_avg_auc = 0.0
                            if task == 'FORECASTING_EVENT':
                                method_avg_f1 = same_length_df[
                                    'Pred_Labels_F1Score'].mean() if 'Pred_Labels_F1Score' in same_length_df.columns and not same_length_df.empty else 0
                                method_avg_auc = same_length_df[
                                    'Pred_Labels_AUC'].mean() if 'Pred_Labels_AUC' in same_length_df.columns and not same_length_df.empty else 0

                            # 获取HistLen值，如果DataFrame中有HistLen列则使用，否则使用默认值
                            HistLen = method_df['HistLen'].iloc[0] if 'HistLen' in method_df.columns and not method_df.empty else 0
                            
                            data_entry = {
                                'Index': idx,
                                'Task': task,
                                'Dataset': dataset,
                                'Method': method,
                                'HistLen': HistLen,
                                'PredLen': int(pred_len),
                                'Avg_Pred_Labels_Accuracy': method_avg_accuracy,
                                'Avg_Pred_Series_MAE': method_avg_mae if task != 'FORECASTING_EVENT' else 0.0,
                                'Total_Samples': len(method_df),
                                'LogTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # 为所有任务添加F1和AUC列
                            if task == 'FORECASTING_EVENT':
                                data_entry['Avg_Pred_Labels_F1Score'] = method_avg_f1
                                data_entry['Avg_Pred_Labels_AUC'] = method_avg_auc
                            else:
                                # 为其他任务（UNDERSTANDING、REASONING、FORECASTING_NUM）设置F1Score和AUC为0.0
                                data_entry['Avg_Pred_Labels_F1Score'] = 0.0
                                data_entry['Avg_Pred_Labels_AUC'] = 0.0
                            
                            all_data.append(data_entry)
                            idx = idx + 1
                    else:
                        # 如果没有Task列，按原来的逻辑处理
                        for method in dataset_methods:
                            method_df = dataset_df[dataset_df['Method'] == method].copy()
                            
                            if method_df.empty:
                                continue
                            
                            # 推断任务类型
                            if method in BASELINE_UNDERSTANDING:
                                task = "UNDERSTANDING"
                            elif method in BASELINE_FORECASTING_NUM:
                                task = "FORECASTING_NUM"
                            elif method in BASELINE_FORECASTING_EVENT:
                                task = "FORECASTING_EVENT"
                            elif method in BASELINE_REASONING:
                                task = "REASONING"
                            else:
                                # 对于OUR_Method，根据数据集推断任务类型
                                if method in OUR_Method:
                                    if dataset in DATASET_UNDERSTANDING:
                                        task = "UNDERSTANDING"
                                    elif dataset in DATASET_FORECASTING_NUM:
                                        task = "FORECASTING_NUM"
                                    elif dataset in DATASET_FORECASTING_EVENT:
                                        task = "FORECASTING_EVENT"
                                    elif dataset in DATASET_REASONING:
                                        task = "REASONING"
                                    else:
                                        task = "Unknown"
                                else:
                                    task = "Unknown"

                            # 标记是否需要更新原始CSV文件
                            csv_updated = False
                            
                            # 为每一行数据进行绘图和指标计算
                            for row_idx, row in method_df.iterrows():
                                # 解析数据
                                pred_series = safe_eval_list(row.get('Pred_Series', '[]'))
                                truth_series = safe_eval_list(row.get('Pred_Series_Truth', '[]'))
                                pred_labels = safe_eval_list(row.get('Pred_Labels', '[]'))
                                truth_labels = safe_eval_list(row.get('Pred_Labels_Truth', '[]'))
                                
                                # 获取原始序列数据（用于绘图）
                                original_series = get_original_series_from_stream(dataset, row_idx - 1)
                                
                                # 为 FORECASTING_NUM 任务生成绘图
                                if task == 'FORECASTING_NUM' and pred_series and truth_series and OUTPUT_PREDICT_IMAGE:
                                    # 构建保存路径
                                    predict_dir = os.path.dirname(file_path)
                                    draw_dir = os.path.join(predict_dir, 'draw')
                                    plot_path = os.path.join(draw_dir, f'{dataset}_{method}_row_{row_idx}.png')
                                    
                                    # 创建绘图
                                    create_forecasting_num_plot(pred_series, truth_series, original_series, row_idx, plot_path, dataset, method)
                                    # print(f"Created plot for {task} task, row {row_idx}: {plot_path}")
                                
                                # 计算扩展指标并更新原始DataFrame
                                if task == 'FORECASTING_NUM':
                                    # 重新计算 MAE
                                    if pred_series and truth_series:
                                        calculated_mae = calculate_mae(pred_series, truth_series)
                                        df.loc[row_idx, 'Pred_Series_MAE'] = calculated_mae
                                        method_df.loc[row_idx, 'Pred_Series_MAE'] = calculated_mae
                                        csv_updated = True
                                
                                elif task == 'REASONING':
                                    # 计算文本相似度
                                    if pred_labels and truth_labels:
                                        text_similarity = calculate_text_similarity(pred_labels, truth_labels)
                                        df.loc[row_idx, 'Pred_Labels_Accuracy'] = text_similarity
                                        method_df.loc[row_idx, 'Pred_Labels_Accuracy'] = text_similarity
                                        csv_updated = True
                                
                                elif task == 'UNDERSTANDING':
                                    # 计算位置匹配准确率
                                    if pred_labels and truth_labels:
                                        position_accuracy = calculate_position_accuracy(pred_labels, truth_labels)
                                        df.loc[row_idx, 'Pred_Labels_Accuracy'] = position_accuracy
                                        method_df.loc[row_idx, 'Pred_Labels_Accuracy'] = position_accuracy
                                        csv_updated = True
                                
                                elif task == 'FORECASTING_EVENT':
                                    # 计算事件预测指标 (F1, AUC, Accuracy) - 不处理Pred_Series
                                    if pred_labels and truth_labels:
                                        # 尝试获取 Impact_Scores
                                        impact_scores = safe_eval_list(row.get('Impact_Scores', '[]'))
                                        
                                        # 计算事件预测指标
                                        event_metrics = calculate_event_metrics(pred_labels, truth_labels, impact_scores)
                                        
                                        # 更新 DataFrame - 不更新Pred_Series_MAE
                                        df.loc[row_idx, 'Pred_Labels_Accuracy'] = event_metrics['accuracy']
                                        method_df.loc[row_idx, 'Pred_Labels_Accuracy'] = event_metrics['accuracy']
                                        
                                        # 添加 F1Score 和 AUC 列（如果不存在）
                                        if 'Pred_Labels_F1Score' not in df.columns:
                                            df['Pred_Labels_F1Score'] = 0.0
                                        if 'Pred_Labels_AUC' not in df.columns:
                                            df['Pred_Labels_AUC'] = 0.0
                                        
                                        df.loc[row_idx, 'Pred_Labels_F1Score'] = event_metrics['f1']
                                        method_df.loc[row_idx, 'Pred_Labels_F1Score'] = event_metrics['f1']
                                        df.loc[row_idx, 'Pred_Labels_AUC'] = event_metrics['auc']
                                        method_df.loc[row_idx, 'Pred_Labels_AUC'] = event_metrics['auc']
                                        
                                        # 确保FORECASTING_EVENT任务的Pred_Series_MAE为0
                                        df.loc[row_idx, 'Pred_Series_MAE'] = 0.0
                                        method_df.loc[row_idx, 'Pred_Series_MAE'] = 0.0
                                        
                                        csv_updated = True
                            
                            # 如果更新了数据，保存回原始CSV文件
                            if csv_updated:
                                df.to_csv(file_path, index=False)
                                # print(f"Updated CSV file: {file_path}")

                        # 计算平均指标
                        method_df = method_df.dropna(subset=['Pred_Series_Len', 'Pred_Series_Truth_Len'])
                        same_length_df = method_df[method_df['Pred_Series_Len'] == method_df['Pred_Series_Truth_Len']]

                        method_avg_accuracy = same_length_df[
                            'Pred_Labels_Accuracy'].mean() if 'Pred_Labels_Accuracy' in same_length_df.columns and not same_length_df.empty else 0
                        method_avg_mae = same_length_df[
                            'Pred_Series_MAE'].mean() if 'Pred_Series_MAE' in same_length_df.columns and not same_length_df.empty else 0
                        
                        # 计算F1和AUC的平均值（仅对FORECASTING_EVENT任务）
                        method_avg_f1 = 0.0
                        method_avg_auc = 0.0
                        if task == 'FORECASTING_EVENT':
                            method_avg_f1 = same_length_df[
                                'Pred_Labels_F1Score'].mean() if 'Pred_Labels_F1Score' in same_length_df.columns and not same_length_df.empty else 0
                            method_avg_auc = same_length_df[
                                'Pred_Labels_AUC'].mean() if 'Pred_Labels_AUC' in same_length_df.columns and not same_length_df.empty else 0

                        data_entry = {
                            'Index': idx,
                            'Task': task,
                            'Dataset': dataset,
                            'Method': method,
                            'HistLen': HistLen ,
                            'PredLen': int(pred_len),
                            'Avg_Pred_Labels_Accuracy': method_avg_accuracy,
                            'Avg_Pred_Series_MAE': method_avg_mae if task != 'FORECASTING_EVENT' else 0.0,
                            'Total_Samples': len(method_df),
                            'LogTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 为FORECASTING_EVENT任务添加F1和AUC列
                        if task == 'FORECASTING_EVENT':
                            data_entry['Avg_Pred_Labels_F1Score'] = method_avg_f1
                            data_entry['Avg_Pred_Labels_AUC'] = method_avg_auc
                        
                        all_data.append(data_entry)
                        idx = idx + 1

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue

    # 创建结果DataFrame
    if all_data:
        result_df = pd.DataFrame(all_data)

        # 确保列的正确顺序，为所有任务添加F1和AUC列
        base_columns = ['Index', 'Task', 'Dataset', 'Method', 'HistLen', 'PredLen', 'Avg_Pred_Labels_Accuracy', 'Avg_Pred_Series_MAE']
        
        # 总是添加F1和AUC列，对于不需要的任务会填充0.0
        base_columns.extend(['Avg_Pred_Labels_F1Score', 'Avg_Pred_Labels_AUC'])
        
        column_order = base_columns + ['Total_Samples', 'LogTime']
        
        # 只选择存在的列
        existing_columns = [col for col in column_order if col in result_df.columns]
        result_df = result_df[existing_columns]

        # 确保排序列中的数据类型正确，避免unhashable type错误
        for col in ['Task', 'Dataset', 'Method', 'HistLen', 'PredLen']:
            if col in result_df.columns:
                # 将列表类型转换为字符串，确保可以排序
                result_df[col] = result_df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
        
        # 按照数据集、历史长度和预测长度排序
        result_df = result_df.sort_values(by=['Task', 'Dataset', 'Method', 'HistLen', 'PredLen'])
        
        # 重新分配Index以确保升序
        result_df['Index'] = range(1, len(result_df) + 1)

        # 增量保存到CSV，如果文件已存在则追加数据
        if os.path.exists(LOG_EXP_METRICS_PATH):
            # 读取现有数据并追加新数据
            existing_df = pd.read_csv(LOG_EXP_METRICS_PATH)
            
            # 确保现有数据也有正确的列顺序和Task列
            if 'Task' not in existing_df.columns:
                # 如果现有文件缺少Task列，需要添加默认值或从其他信息推断
                existing_df['Task'] = 'Unknown'  # 或者根据Method来推断Task类型
            
            # 如果现有数据没有Index列，添加Index列
            if 'Index' not in existing_df.columns:
                existing_df['Index'] = range(1, len(existing_df) + 1)
            
            # 确保现有数据有相同的列顺序
            existing_columns = existing_df.columns.tolist()
            for col in column_order:
                if col not in existing_columns:
                    existing_df[col] = None
            existing_df = existing_df[column_order]
            
            # 检查是否有完全相同的记录
            merged_df = pd.concat([existing_df, result_df])

            # 去重：保留最后出现的记录
            merged_df = merged_df.drop_duplicates(subset=['Task', 'Dataset', 'Method', 'HistLen', 'PredLen'],
                                                  keep='last')

            # 确保排序列中的数据类型正确，避免unhashable type错误
            for col in ['Task', 'Dataset', 'Method', 'HistLen', 'PredLen']:
                if col in merged_df.columns:
                    # 将列表类型转换为字符串，确保可以排序
                    merged_df[col] = merged_df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            
            # 按照Task、Dataset、Method、HistLen、PredLen排序
            merged_df = merged_df.sort_values(by=['Task', 'Dataset', 'Method', 'HistLen', 'PredLen'])
            
            # 重新分配Index以确保升序
            merged_df['Index'] = range(1, len(merged_df) + 1)

            # 保存回文件
            merged_df.to_csv(LOG_EXP_METRICS_PATH, index=False)
        else:
            result_df.to_csv(LOG_EXP_METRICS_PATH, index=False)

        print(f"Analysis results saved to {LOG_EXP_METRICS_PATH}")
        
        # 设置pandas显示选项以确保完整打印
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.max_rows', None)     # 显示所有行
        pd.set_option('display.width', None)        # 不限制显示宽度
        pd.set_option('display.max_colwidth', None) # 不限制列宽
        
        # print("Complete DataFrame contents:")
        # print(result_df)
        
        # 恢复默认设置（可选）
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
    else:
        print("No valid data found")


def generate_rq1_csv():
    """
    生成RQ_1.csv文件，包含各方法在不同任务上的性能指标
    """
    import os
    import pandas as pd
    import glob
    
    # 确保results目录存在
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 读取TSQA数据
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)
    tsqa_file = os.path.join(_project_root, "datasets/TSQA/predict-TSQA/predict_compare.csv")
    qats4_file = os.path.join(_project_root, "datasets/QATS-4/predict-QATS-4/predict_compare.csv")
    metrics_file = os.path.join(_project_root, "output/metrics_call_llm.csv")
    
    rq1_data = []

    if os.path.exists(tsqa_file):
        tsqa_df = pd.read_csv(tsqa_file)
        
        # 按Method分组处理TSQA数据
        for method, group in tsqa_df.groupby('Method'):
            # 确定Type
            if '(+v)' in method:
                type_val = '+TS2Vision'
            else:
                type_val = 'TS2Text'
            
            # 计算Trend, Season, Volatility, Outlier准确率
            trend_acc = season_acc = volatility_acc = outlier_acc = 0.0
            trend_sum = season_sum = volatility_sum = outlier_sum = 0.0
            
            # 遍历每一行数据进行准确率计算
            for _, row in group.iterrows():
                try:
                    pred_labels = safe_eval_list(row['Pred_Labels']) if pd.notna(row['Pred_Labels']) else []
                    truth_labels = safe_eval_list(row['Pred_Labels_Truth']) if pd.notna(row['Pred_Labels_Truth']) else []
                    
                    # 检查Truth_Labels中是否包含关键词，计算xxx_sum（分母）
                    truth_labels_str = str(truth_labels).lower()
                    if 'trend' in truth_labels_str:
                        trend_sum += 1
                    if 'season' in truth_labels_str:
                        season_sum += 1
                    if 'volatility' in truth_labels_str:
                        volatility_sum += 1
                    if 'outlier' in truth_labels_str:
                        outlier_sum += 1
                    
                    # 使用相似度计算而不是完全匹配
                    similarity = calculate_text_similarity_fallback(str(pred_labels), str(truth_labels))
                    if similarity >= TEXT_SIMILARITY_THRESHOLD:  # 相似度阈值
                        if 'trend' in truth_labels_str:
                            trend_acc += 1
                        if 'season' in truth_labels_str:
                            season_acc += 1
                        if 'volatility' in truth_labels_str:
                            volatility_acc += 1
                        if 'outlier' in truth_labels_str:
                            outlier_acc += 1
                            
                except Exception as e:
                    continue
            
            # 计算最终准确率
            trend_acc = trend_acc / trend_sum if trend_sum > 0 else 0.0
            season_acc = season_acc / season_sum if season_sum > 0 else 0.0
            volatility_acc = volatility_acc / volatility_sum if volatility_sum > 0 else 0.0
            outlier_acc = outlier_acc / outlier_sum if outlier_sum > 0 else 0.0
            
            rq1_data.append({
                'Method': method,
                'Type': type_val,
                'Trend': trend_acc,
                'Season': season_acc,
                'Volatility': volatility_acc,
                'Outlier': outlier_acc,
                'Induct.': 0.0,  # 将从QATS-4数据计算
                'Deduct.': 0.0,  # 将从QATS-4数据计算
                'Causal': 0.0,   # 将从QATS-4数据计算
                'MCQ2': 0.0,     # 将从QATS-4数据计算
                'Overall': 0.0,  # 将计算8个指标的平均值
                'TTFT': 0.0,     # 将从metrics文件计算
                'Tokens': 0.0,   # 将从metrics文件计算
                'Cost': 0.0      # 将从metrics文件计算
            })

    if os.path.exists(qats4_file):
        qats4_df = pd.read_csv(qats4_file)
        
        for i, row in enumerate(rq1_data):
            method = row['Method']
            method_data = qats4_df[qats4_df['Method'] == method]
            
            if len(method_data) > 0:
                # 修复：正确计算QATS-4数据的准确率
                induct_acc = deduct_acc = causal_acc = mcq2_acc = 0.0
                induct_sum = deduct_sum = causal_sum = mcq2_sum = 0.0
                
                # 遍历每一行数据进行准确率计算
                for _, qats_row in method_data.iterrows():
                    try:
                        pred_labels = safe_eval_list(qats_row['Pred_Labels']) if pd.notna(qats_row['Pred_Labels']) else []
                        truth_labels = safe_eval_list(qats_row['Pred_Labels_Truth']) if pd.notna(qats_row['Pred_Labels_Truth']) else []
                        
                        # 检查Truth_Labels中是否包含关键词，计算xxx_sum（分母）
                        truth_labels_str = str(truth_labels).lower()
                        if 'induct' in truth_labels_str:
                            induct_sum += 1
                        if 'deduct' in truth_labels_str:
                            deduct_sum += 1
                        if 'causal' in truth_labels_str:
                            causal_sum += 1
                        if 'mcq' in truth_labels_str:
                            mcq2_sum += 1
                        
                        # 计算相似度作为准确率
                        if pred_labels and truth_labels:
                            similarity = calculate_text_similarity(pred_labels, truth_labels)
                            
                            if 'induct' in truth_labels_str:
                                induct_acc += similarity
                            if 'deduct' in truth_labels_str:
                                deduct_acc += similarity
                            if 'causal' in truth_labels_str:
                                causal_acc += similarity
                            if 'mcq' in truth_labels_str:
                                mcq2_acc += similarity
                                
                    except Exception as e:
                        continue
                
                # 计算最终准确率
                rq1_data[i]['Induct.'] = induct_acc / induct_sum if induct_sum > 0 else 0.0
                rq1_data[i]['Deduct.'] = deduct_acc / deduct_sum if deduct_sum > 0 else 0.0
                rq1_data[i]['Causal'] = causal_acc / causal_sum if causal_sum > 0 else 0.0
                rq1_data[i]['MCQ2'] = mcq2_acc / mcq2_sum if mcq2_sum > 0 else 0.0

    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
        reasoning_understanding = metrics_df[metrics_df['Task'].isin(['REASONING', 'UNDERSTANDING'])]
        
        for i, row in enumerate(rq1_data):
            method = row['Method']
            method_metrics = reasoning_understanding[reasoning_understanding['Method'] == method]
            
            if len(method_metrics) > 0:
                rq1_data[i]['TTFT'] = method_metrics['TTFT'].mean()
                rq1_data[i]['Tokens'] = (method_metrics['InputTokens'] + method_metrics['OutputTokens']).mean()
                rq1_data[i]['Cost'] = method_metrics['Cost'].mean()
    
    # 计算Overall（8个指标的平均值）
    for i, row in enumerate(rq1_data):
        eight_metrics = [row['Trend'], row['Season'], row['Volatility'], row['Outlier'],
                        row['Induct.'], row['Deduct.'], row['Causal'], row['MCQ2']]
        rq1_data[i]['Overall'] = sum(eight_metrics) / len(eight_metrics) if any(eight_metrics) else 0.0
    
    # 创建DataFrame并保存
    rq1_df = pd.DataFrame(rq1_data)
    rq1_csv_path = os.path.join(results_dir, "RQ_1.csv")
    rq1_df.to_csv(rq1_csv_path, index=False)
    
    # print(f"RQ_1.csv has been generated: {rq1_csv_path}")
    return rq1_csv_path


def generate_rq2_csv():
    """
    生成RQ_2.csv文件，包含不同数据集上各方法的预测性能
    """
    import os
    import pandas as pd
    
    # 确保results目录存在
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    rq2_data = []
    
    for dataset in DATASET_FORECASTING_NUM:
        predict_file = f"../datasets/{dataset}/predict-{dataset}/predict_compare.csv"
        
        if os.path.exists(predict_file):
            df = pd.read_csv(predict_file)
            
            # 按Method分组计算平均MAE
            method_maes = df.groupby('Method')['Pred_Series_MAE'].mean()
            
            # 获取Hist/Pred字段值
            hist_pred_value = ''
            if not df.empty and 'HistLen' in df.columns and 'Pred_Series_Len' in df.columns:
                # 取第一行的HistLen和Pred_Series_Len来构造Hist/Pred字段
                first_row = df.iloc[0]
                hist_len = first_row.get('HistLen', '')
                pred_len = first_row.get('Pred_Series_Len', '')
                if pd.notna(hist_len) and pd.notna(pred_len):
                    hist_pred_value = f"{hist_len}/{pred_len}"
            
            row_data = {'Dataset': dataset, 'Hist/Pred': hist_pred_value}
            
            # 为每个方法设置对应的列
            for method, mae in method_maes.items():
                if 'PromptCast' in method:
                    row_data['PromptCast'] = mae
                elif 'TimeCP' in method:
                    row_data['TimeCP'] = mae
                elif 'StreamTS-Agents' in method:
                    row_data['StreamTS-Agents'] = mae
            
            # 确保所有列都存在
            for col in ['PromptCast', 'TimeCP', 'StreamTS-Agents']:
                if col not in row_data:
                    row_data[col] = 0.0
            
            rq2_data.append(row_data)
        else:
            # 如果文件不存在，创建空行
            rq2_data.append({
                'Dataset': dataset,
                'Hist/Pred': '',
                'PromptCast': 0.0,
                'TimeCP': 0.0,
                'StreamTS-Agents': 0.0
            })
    
    # 创建DataFrame并保存
    rq2_df = pd.DataFrame(rq2_data)
    rq2_csv_path = os.path.join(results_dir, "RQ_3.csv")
    rq2_df.to_csv(rq2_csv_path, index=False)
    
    # print(f"RQ_3.csv has been generated: {rq2_csv_path}")
    return rq2_csv_path
def generate_rq3_csv():
    """
    生成RQ_3.csv文件，包含事件预测任务上各方法的F1和AUC性能
    """
    import os
    import pandas as pd
    
    # 确保results目录存在
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    rq3_data = []
    
    for dataset in DATASET_FORECASTING_EVENT:
        predict_file = f"../datasets/{dataset}/predict-{dataset}/predict_compare.csv"
        
        if os.path.exists(predict_file):
            df = pd.read_csv(predict_file)
            
            # 按Method分组计算平均F1和AUC
            method_metrics = {}
            for method in df['Method'].unique():
                method_df = df[df['Method'] == method]
                
                # 计算平均F1（如果列存在）
                avg_f1 = method_df['Pred_Labels_F1'].mean() if 'Pred_Labels_F1' in method_df.columns else 0.0
                # 计算平均AUC（如果列存在）
                avg_auc = method_df['Pred_Labels_AUC'].mean() if 'Pred_Labels_AUC' in method_df.columns else 0.0
                # 计算平均Accuracy
                avg_accuracy = method_df['Pred_Labels_Accuracy'].mean() if 'Pred_Labels_Accuracy' in method_df.columns else 0.0
                
                method_metrics[method] = {
                    'f1': avg_f1,
                    'auc': avg_auc,
                    'accuracy': avg_accuracy
                }
            
            # 获取Hist/Pred字段值
            hist_pred_value = ''
            if not df.empty and 'HistLen' in df.columns and 'Pred_Series_Len' in df.columns:
                first_row = df.iloc[0]
                hist_len = first_row.get('HistLen', '')
                pred_len = first_row.get('Pred_Series_Len', '')
                if pd.notna(hist_len) and pd.notna(pred_len):
                    hist_pred_value = f"{hist_len}/{pred_len}"
            
            row_data = {'Dataset': dataset, 'Hist/Pred': hist_pred_value}
            
            # 为每个方法设置对应的列
            for method, metrics in method_metrics.items():
                if 'TimeCAP' in method:
                    row_data['TimeCAP_F1'] = metrics['f1']
                    row_data['TimeCAP_AUC'] = metrics['auc']
                    row_data['TimeCAP_Accuracy'] = metrics['accuracy']
                elif 'StreamTS-Agents' in method:
                    row_data['StreamTS-Agents_F1'] = metrics['f1']
                    row_data['StreamTS-Agents_AUC'] = metrics['auc']
                    row_data['StreamTS-Agents_Accuracy'] = metrics['accuracy']
            
            # 确保所有列都存在
            for method_prefix in ['TimeCAP', 'StreamTS-Agents']:
                for metric_suffix in ['_F1', '_AUC', '_Accuracy']:
                    col_name = method_prefix + metric_suffix
                    if col_name not in row_data:
                        row_data[col_name] = 0.0
            
            rq3_data.append(row_data)
        else:
            # 如果文件不存在，创建空行
            row_data = {
                'Dataset': dataset,
                'Hist/Pred': '',
                'TimeCAP_F1': 0.0,
                'TimeCAP_AUC': 0.0,
                'TimeCAP_Accuracy': 0.0,
                'StreamTS-Agents_F1': 0.0,
                'StreamTS-Agents_AUC': 0.0,
                'StreamTS-Agents_Accuracy': 0.0
            }
            rq3_data.append(row_data)
    
    # 创建DataFrame并保存
    rq3_df = pd.DataFrame(rq3_data)
    rq3_csv_path = os.path.join(results_dir, "RQ_2.csv")
    rq3_df.to_csv(rq3_csv_path, index=False)
    
    # print(f"RQ_2.csv has been generated: {rq3_csv_path}")
    return rq3_csv_path


if __name__ == "__main__":
    analyze_predict_compare_results()
    
    # 生成RQ_1.csv、RQ_2.csv和RQ_3.csv
    generate_rq1_csv()
    generate_rq2_csv()
    generate_rq3_csv()