import csv
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.common import *
from utils.config import *
from preprocess.variable_utils import get_dataset_variables

def process_event_dataset(dataset_path):

    # 判断传入的是文件路径还是目录路径
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
            writer.writerow(['Index', 'Dataset', 'Variable', 'Positions', 'Labels', 'Series'])

    # 读取数据集
    df = pd.read_csv(input_path)
    
    # 提取标签列
    labels = None
    if 'label' in df.columns:
        labels = df['label'].tolist()
        df = df.drop('label', axis=1)
    elif 'Label' in df.columns:
        labels = df['Label'].tolist()
        df = df.drop('Label', axis=1)
    else:
        raise ValueError(f"No label column found in {input_path}")
    
    # 如果存在date列，则移除它
    if 'date' in df.columns:
        df = df.drop('date', axis=1)

    # 为每个列生成时间序列图表（排除label列，因为label已经被提取并移除）
    for column in tqdm(df.columns, desc=f"Generating {dataset_name} streaming samples"):
        if OUTPUT_DATASET_IMAGE:
            filename = get_filename(column)
            title = column.split('(')[0].strip() if '(' in column else column
            # 创建图表
            plt.figure(figsize=(25, 4))
            plt.plot(range(len(df)), normalize_series(df[column].tolist()), color='#1f77b4', linewidth=1.2)
            plt.title(title, pad=20)  # 增加标题与图形的间距
            plt.grid(alpha=0.3)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)  # 手动调整边距
            # 保存图表
            plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=150)
            plt.close()
    
    # 准备输出数据
    output_data = []
    
    # 检查是否为FORECASTING_EVENT任务且数据集在指定列表中
    is_forecasting_event_dataset = ("FORECASTING_EVENT" in TASK and dataset_name in DATASET_FORECASTING_EVENT)
    
    if is_forecasting_event_dataset:
        # 对于FORECASTING_EVENT任务的特定数据集，使用新的标签分段逻辑
        try:
            positions_info, label_descriptions = analyze_label_positions(input_path)
        except Exception as e:
            print(f"Warning: Failed to analyze label positions for {dataset_name}: {e}")
            # 如果分析失败，回退到空列表
            positions_info = []
            label_descriptions = []
    else:
        # 直接使用原始标签，不进行数值映射
        processed_labels = labels
        
        # 合并相同的连续标签，生成对应的位置区间
        merged_positions = []
        merged_labels = []
        
        if processed_labels:
            current_label = processed_labels[0]
            start_pos = 0
            
            for i in range(1, len(processed_labels)):
                # 如果标签发生变化，记录前一个区间
                if processed_labels[i] != current_label:
                    merged_positions.append((start_pos, i - 1))
                    merged_labels.append(str(current_label))
                    current_label = processed_labels[i]
                    start_pos = i
            
            # 添加最后一个区间
            merged_positions.append((start_pos, len(processed_labels) - 1))
            merged_labels.append(str(current_label))
        
        positions_info = merged_positions
        label_descriptions = merged_labels
    
    # 获取数据集变量信息
    variables = get_dataset_variables(dataset_name, input_path)
    # 将变量字符串分割为列表
    variable_list = [v.strip() for v in variables.split(',')] if variables else []
    
    for idx, col in enumerate(df.columns):
        # 对于事件预测，标签对应每个时间点，所有时间序列变量共享相同的标签序列
        series_data = normalize_series(df[col].tolist())
        
        # 获取对应的变量名，如果没有则使用列名
        variable_name = variable_list[idx] if idx < len(variable_list) else col
        
        output_data.append({
            'Index': idx + 1,
            'Dataset': dataset_name,
            'Variable': variable_name,
            'Positions': str(positions_info),
            'Labels': str(label_descriptions),
            'Series': str(series_data)
        })
    
    # 保存到 CSV 文件
    pd.DataFrame(output_data).to_csv(output_path, index=False)
    
    return f"Processed {dataset_name}: {len(df.columns)} series"


def analyze_label_positions(csv_file_path):
    """
    分析CSV文件中标签的位置分布，生成position和labels列表
    tuple: (positions, labels) 其中positions是[(start, end), ...]格式，labels是对应的标签列表
    """
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 获取标签列
    if 'label' in df.columns:
        labels = df['label'].tolist()
    elif 'Label' in df.columns:
        labels = df['Label'].tolist()
    else:
        raise ValueError("No label column found in the CSV file")

    # 获取数据集名称以确定处理逻辑
    dataset_name = os.path.basename(os.path.dirname(csv_file_path))

    positions = []
    position_labels = []

    # 获取所有有标签的行位置
    label_indices = []
    label_values = []
    for i, label in enumerate(labels):
        if pd.notna(label) and label != '':
            label_indices.append(i + 1)  # +1转换为1-based行号
            label_values.append(label)
    
    if not label_indices:
        return positions, position_labels
    
    # 根据数据集类型使用不同的分段逻辑
    if dataset_name.startswith('Weather_'):
        # Weather数据集：使用原有逻辑，每个标签覆盖到下一个标签前
        i = 0
        while i < len(labels):
            if pd.notna(labels[i]) and labels[i] != '':
                start_pos = i + 1
                current_label = labels[i]
                
                # 查找下一个有标签的行
                j = i + 1
                while j < len(labels) and (pd.isna(labels[j]) or labels[j] == ''):
                    j += 1
                
                # 结束位置
                if j < len(labels):
                    end_pos = j
                else:
                    end_pos = len(labels)
                
                positions.append((start_pos, end_pos))
                position_labels.append(current_label)
                i = j
            else:
                i += 1
                
    elif dataset_name.startswith('Healthcare_') or dataset_name.startswith('Finance_'):
        # Healthcare和Finance数据集：每一段都从1到第一个标签行
        if label_indices:
            # 每个标签都对应一个从1到该标签行的区间
            for i in range(len(label_indices)):
                label_pos = label_indices[i]
                positions.append((1, label_pos))
                position_labels.append(label_values[i])
    else:
        # 默认逻辑：使用Weather的方式
        i = 0
        while i < len(labels):
            if pd.notna(labels[i]) and labels[i] != '':
                start_pos = i + 1
                current_label = labels[i]
                
                j = i + 1
                while j < len(labels) and (pd.isna(labels[j]) or labels[j] == ''):
                    j += 1
                
                if j < len(labels):
                    end_pos = j
                else:
                    end_pos = len(labels)
                
                positions.append((start_pos, end_pos))
                position_labels.append(current_label)
                i = j
            else:
                i += 1

    return positions, position_labels
