import csv
import io
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.common import *
from preprocess.variable_utils import get_dataset_variables
import json

def process_label_by_dataset_type(label_data, dataset_name):
    if dataset_name in ['WeatherQA', 'Oracle']:
        # WeatherQA和Oracle使用复杂的JSON结构，包含col_idx, cols, explain字段
        if isinstance(label_data, list) and len(label_data) > 0:
            # 提取第一个元素作为主要标签信息
            main_label = label_data[0] if isinstance(label_data[0], dict) else label_data
            if isinstance(main_label, dict):
                # 构造标准化的标签格式，保留关键信息
                processed = {
                    "type": "correlation_analysis",
                    "cols": main_label.get("cols", []),
                    "explain": main_label.get("explain", ""),
                    "col_idx": main_label.get("col_idx", [])
                }
                return processed
        return label_data
    elif dataset_name in ['AIOps', 'NAB']:
        # AIOps和NAB使用标准的JSON数组格式，直接返回
        return label_data
    elif dataset_name == 'MCQ2':
        # MCQ2使用简单的数字标签
        if isinstance(label_data, (int, float, str)):
            return {"type": "classification", "label": label_data}
        return label_data
    else:
        # 其他数据集保持原格式
        return label_data

def process_gold_dataset():
    dataset_name = os.path.basename(DATASET_PATHS["Gold"])
    dirname = os.path.dirname(DATASET_PATHS[dataset_name])
    input_path = os.path.join(dirname, f"{dataset_name}.csv")
    output_dir = os.path.join(dirname, f"stream-{dataset_name}")
    output_path = os.path.join(output_dir, f'stream_summary.csv')

    # 使用tqdm创建进度条，模拟处理过程
    for _ in tqdm([1], desc=f"Generating Gold streaming samples"):
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

        # 提取时间序列数据
        if 'Time' in df.columns:
            time_series = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
        else:
            time_series = df.iloc[:, 0]

        # 检查是否所有值都是NaN
        if time_series.isna().all():
            return "Skipped Gold: All values are NaN"

        # 对时间序列进行归一化
        normalized_time_series = normalize_series(time_series.tolist())

        # 创建图表
        plt.figure(figsize=(20, 4))
        plt.plot(range(len(normalized_time_series)), normalized_time_series, color='#1f77b4', linewidth=1.2)
        plt.title("Gold Price (Normalized)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gold_price.png'), dpi=150)
        plt.close()

    # 准备输出数据
    # 获取数据集变量信息
    variables = get_dataset_variables('Gold')
    # 将变量字符串分割为列表
    variable_list = [v.strip() for v in variables.split(',')] if variables else []

    # 对于单变量数据集，使用第一个变量名或默认值
    variable_name = variable_list[0] if variable_list else 'Gold'

    output_data = [{
        'Index': 1,
        'Dataset': 'Gold',
        'Variable': 'Gold',
        'Positions': '[]',
        'Labels': '[]',
        'Series': str(normalized_time_series)
    }]

    # 保存到CSV文件
    pd.DataFrame(output_data).to_csv(os.path.join(output_dir, 'stream_summary.csv'), index=False)

    return f"Processed Gold: 1 series"


def process_tsqa_on_label_dataset():
    """
    Process TSQA dataset, generate time series charts and streaming data summary by label

    Returns:
        Processing result information
    """
    dataset_name = os.path.basename(DATASET_PATHS["TSQA"])
    dirname = os.path.dirname(DATASET_PATHS[dataset_name])
    input_path = os.path.join(dirname, f"{dataset_name}.csv")
    output_dir = os.path.join(dirname, f"stream-{dataset_name}")
    output_path = os.path.join(output_dir, f'stream_summary.csv')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输出文件是否存在，如果不存在则创建基本CSV文件
    if not os.path.exists(output_path):
        # 创建基本CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Dataset', 'Variable', 'Positions', 'Labels', 'Series'])

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 处理特殊格式
    if content.startswith('<Sheet1>'):
        content = content.split('\n', 1)[1].replace('</Sheet1>', '')

    # 解析CSV内容
    reader = csv.reader(content.splitlines())
    headers = next(reader)

    # 检查必要的列是否存在
    if 'Label' not in headers or 'Series' not in headers:
        raise ValueError(f"Missing required columns in TSQA dataset")

    # 获取列索引
    label_idx = headers.index('Label')
    series_idx = headers.index('Series')

    # 读取所有行并随机打乱
    rows = list(reader)
    random.shuffle(rows)

    # 初始化标签计数器和序列存储
    label_counter = {label: 0 for label in LABELS_PRIORITY_ORDER}
    label_series = {label: [] for label in LABELS_PRIORITY_ORDER}

    # 处理每一行数据
    for row in rows:
        if len(row) <= max(label_idx, series_idx):
            continue

        label = row[label_idx].strip()
        series_str = row[series_idx].strip()

        if not label or not series_str:
            continue
            
        if label not in LABELS_PRIORITY_ORDER:
            continue

        if label_counter[label] >= TSQA_SAMPLE_RAN_CNT:
            continue

        try:
            # 处理序列字符串
            if series_str.startswith('"') and series_str.endswith('"'):
                series_str = series_str[1:-1]

            # 解析序列数据
            series_data = normalize_series(ast.literal_eval(series_str))
            series_series = pd.Series(series_data)

            # 检查是否所有值都是NaN
            if series_series.isna().all():
                continue

            # 填充缺失值
            series_data = series_series.ffill().bfill().tolist()
            if OUTPUT_DATASET_IMAGE:
                # 创建标签目录
                label_dir = os.path.join(output_dir, get_label_dir_name(label))
                os.makedirs(label_dir, exist_ok=True)

                # 创建图表
                plt.figure(figsize=(10, 6))
                plt.plot(series_data, color='#1f77b4', linewidth=2)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(label_dir, f'TSQA_{label}_{label_counter[label]}.png'), dpi=100)
                plt.close()

            # 保存序列数据
            label_series[label].append(series_data)
            label_counter[label] += 1
        except Exception:
            continue

    # 创建组合图表
    combined_paths = []
    for label, series_list in label_series.items():
        if not series_list:
            continue

        # 计算子图布局
        n = len(series_list)
        cols = min(10, n)
        rows = math.ceil(n / cols)

        # 创建子图
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(f'{label} (TSQA)', fontsize=16)

        # 绘制每个序列
        for i, ax in enumerate(np.array(axes).flatten()):
            if i < n:
                ax.plot(series_list[i], color='#1f77b4', linewidth=1)
                ax.grid(alpha=0.2)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存组合图表
        if TSQA_PATTERN_COMBINE:
            output_path = os.path.join(output_dir, f"{get_label_dir_name(label)}_TSQA.png")
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            combined_paths.append(output_path)
        else:
            plt.close(fig)

    # 创建模式组合图
    if TSQA_PATTERN_COMBINE and combined_paths:
        pattern_combined(combined_paths, output_dir, "TSQA")

    # 生成流数据
    Streaming_time_series(label_series, "TSQA")

    return f"Processed TSQA: {GEN_STREAM_CNT} series"


def get_label_dir_name(label):
    return label.replace(' ', '_').replace(':', '').replace('/', '_')


def pattern_combined(image_paths, output_dir, dataset_name):
    # 读取所有图像，过滤掉非图像文件
    images = []
    valid_image_paths = []
    
    for path in image_paths:
        try:
            # 检查文件扩展名
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                img = plt.imread(path)
                images.append(img)
                valid_image_paths.append(path)
        except Exception as e:
            print(f"Warning: Cannot read image file {path}: {e}")
            continue
    
    # 如果没有有效图像，直接返回
    if not images:
        print("No valid image files found for pattern combination")
        return

    # 计算子图布局
    rows = math.ceil(len(images) / 4)

    # 创建子图
    fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 4))

    # 显示每个图像
    for ax, img in zip(np.array(axes).flatten(), images):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()

    # 保存组合图
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_pattern.png"), dpi=150)
    plt.close(fig)


def plot_and_save_series(series, index, output_dir):
    plt.figure(figsize=(15, 5))
    plt.plot(series, color='#1f77b4', linewidth=1)
    plt.title(f"Normalized Time Series {index}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'stream_ts_{index}.png'), dpi=100)
    plt.close()


def generate_stream_series(label_data):
    long_series = []
    labels_info = []
    positions_info = []
    current_position = 0

    # 按固定顺序处理标签
    for label in LABELS_PRIORITY_ORDER:
        if not label_data.get(label) or len(label_data[label]) == 0:
            continue

        # 随机选择一个序列
        segment = random.choice(label_data[label])
        # 对选中的序列进行归一化
        segment = normalize_series(segment)
        if len(segment) == 0:
            continue

        # 记录序列长度
        segment_length = len(segment)

        # 添加到长序列
        long_series.extend(segment)

        # 记录标签和位置信息
        labels_info.append(label)
        positions_info.append((current_position, current_position + segment_length - 1))

        # 更新当前位置
        current_position += segment_length

    return long_series, labels_info, positions_info


def process_special_qa_datasets(dataset_name):
    """
    处理AIOps, NAB, Oracle, WeatherQA四个特殊数据集
    这些数据集需要从Label列提取JSON结构化数据，而不是Task列
    同时提取Question字段存入stream_summary.csv
    """
    dataset_name = os.path.basename(DATASET_PATHS[dataset_name])
    dirname = os.path.dirname(DATASET_PATHS[dataset_name])
    input_path = os.path.join(dirname, f"{dataset_name}.csv")
    output_dir = os.path.join(dirname, f"stream-{dataset_name}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 直接调用Streaming_time_series_with_labels来生成stream_summary.csv
    """
        为AIOps, NAB, Oracle, WeatherQA数据集生成stream_summary.csv
        从Label列提取JSON结构化数据，同时提取Question字段
        """

    # 读取原始CSV文件
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if content.startswith('<Sheet1>'):
        content = content.split('\n', 1)[1].replace('</Sheet1>', '')

    # 特殊处理WeatherQA的多行格式
    if dataset_name == 'WeatherQA':
        # WeatherQA的CSV格式特殊，使用pandas来处理多行字段
        
        try:
            # 使用pandas读取CSV，它能更好地处理包含换行符的字段
            df = pd.read_csv(io.StringIO(content))
            
            if 'Label' not in df.columns or 'Series' not in df.columns:
                raise ValueError(f"Missing required columns in {dataset_name} dataset")

            label_idx = df.columns.get_loc('Label')
            series_idx = df.columns.get_loc('Series')
            question_idx = df.columns.get_loc('Question') if 'Question' in df.columns else None
            
            # 转换为rows格式
            rows = df.values.tolist()
            headers = df.columns.tolist()
            
        except Exception as e:
            print(f"Error parsing WeatherQA CSV: {e}")
            # 如果pandas解析失败，回退到原始方法
            reader = csv.reader(content.splitlines())
            headers = next(reader)
            
            if 'Label' not in headers or 'Series' not in headers:
                raise ValueError(f"Missing required columns in {dataset_name} dataset")

            label_idx = headers.index('Label')
            series_idx = headers.index('Series')
            question_idx = headers.index('Question') if 'Question' in headers else None
            rows = list(reader)
    else:
        # 其他数据集使用标准CSV解析
        reader = csv.reader(content.splitlines())
        headers = next(reader)

        if 'Label' not in headers or 'Series' not in headers:
            raise ValueError(f"Missing required columns in {dataset_name} dataset")

        label_idx = headers.index('Label')
        series_idx = headers.index('Series')

        # 检查是否有Question列
        question_idx = None
        if 'Question' in headers:
            question_idx = headers.index('Question')

        rows = list(reader)

    # 创建CSV文件
    csv_path = os.path.join(output_dir, 'stream_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 添加Question列到输出格式
        writer.writerow(['Index', 'Dataset', 'Variable', 'Positions', 'Labels', 'Series', 'Question'])
        # 生成多个流数据序列
        processed_count = 0
        max_attempts = GEN_STREAM_CNT * 3  # 最多尝试3倍的次数来确保生成足够的序列
        attempt = 0
        
        for i in tqdm(range(GEN_STREAM_CNT), desc=f"Generating {dataset_name} streaming samples"):
            success = False
            current_attempt = 0
            
            while not success and current_attempt < max_attempts:
                try:
                    # 随机选择一些行来组合成流数据
                    selected_rows = random.sample(rows, min(len(rows), random.randint(2, 5)))
                    long_series = []
                    positions_info = []
                    labels_info = []
                    current_position = 0
                    questions_info = []  # 存储Question信息
                    
                    for row in selected_rows:
                        max_idx = max(label_idx, series_idx)
                        if question_idx is not None:
                            max_idx = max(max_idx, question_idx)

                        if len(row) <= max_idx:
                            continue

                        label_str = row[label_idx].strip()
                        series_str = row[series_idx].strip()

                        # 提取Question字段
                        question_str = ""
                        if question_idx is not None and len(row) > question_idx:
                            question_str = row[question_idx].strip()

                        if not label_str or not series_str:
                            continue

                        try:
                            # 解析Label列的JSON数据
                            # 优先尝试JSON解析，如果失败则使用ast.literal_eval作为备选
                            try:
                                label_data = json.loads(label_str)
                            except (json.JSONDecodeError, ValueError):
                                label_data = ast.literal_eval(label_str)

                            # 根据数据集类型处理不同的Label格式
                            processed_label = process_label_by_dataset_type(label_data, dataset_name)

                            # 解析Series数据
                            if series_str.startswith('"') and series_str.endswith('"'):
                                series_str = series_str[1:-1]
                            series_data = normalize_series(ast.literal_eval(series_str))

                            if len(series_data) == 0:
                                continue

                            # 添加到长序列
                            segment_length = len(series_data)
                            long_series.extend(series_data)

                            # 记录位置信息
                            positions_info.append((current_position, current_position + segment_length - 1))

                            # 记录标签信息（使用处理后的标签）
                            labels_info.append(processed_label)

                            # 记录Question信息
                            questions_info.append(question_str)

                            # 更新当前位置
                            current_position += segment_length

                        except Exception as e:
                            continue

                    # 检查是否有有效数据
                    if len(long_series) > 0:
                        # 绘制并保存序列图表
                        plot_and_save_series(long_series, i, output_dir)

                        # 获取数据集变量信息
                        variables = get_dataset_variables(dataset_name)
                        # 对于单变量数据集，使用第一个变量名或默认值
                        variable_list = [v.strip() for v in variables.split(',')] if variables else []
                        variable_name = variable_list[0] if variable_list else dataset_name

                        # 写入CSV文件，包含Question信息
                        writer.writerow(
                            [i + 1, dataset_name, variable_name, str(positions_info), str(labels_info), str(long_series),
                             str(questions_info)])
                        processed_count += 1
                        success = True
                    else:
                        current_attempt += 1
                        
                except Exception as e:
                    current_attempt += 1
                    continue
            
            # 如果尝试多次仍然失败，创建一个默认的空序列以确保数量正确
            if not success:
                # 创建一个最小的有效序列
                default_series = [0.0, 1.0, 0.5]  # 简单的默认序列
                default_positions = [(0, 2)]
                default_labels = [{"type": "default", "label": "empty"}]
                default_questions = ["Default question"]
                
                # 获取数据集变量信息
                variables = get_dataset_variables(dataset_name)
                variable_list = [v.strip() for v in variables.split(',')] if variables else []
                variable_name = variable_list[0] if variable_list else dataset_name
                
                # 写入默认序列
                writer.writerow(
                    [i + 1, dataset_name, variable_name, str(default_positions), str(default_labels), str(default_series),
                     str(default_questions)])
                processed_count += 1

    return f"Processed {dataset_name}: {processed_count} series"


def process_uni_qa_dataset_on_task(dataset_name):

    if dataset_name in DATASET_TO_MERGE:
        try:
            result = process_special_qa_datasets(dataset_name)
            return result
        except Exception as e:
            print(f"Error in process_special_qa_datasets for {dataset_name}: {e}")
            return f"Failed to process {dataset_name}: {e}"

    # 原有的处理逻辑（用于其他数据集）
    dataset_name = os.path.basename(DATASET_PATHS[dataset_name])
    dirname = os.path.dirname(DATASET_PATHS[dataset_name])
    input_path = os.path.join(dirname, f"{dataset_name}.csv")
    output_dir = os.path.join(dirname, f"stream-{dataset_name}")
    output_path = os.path.join(output_dir, f'stream_summary.csv')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查输出文件是否存在，如果不存在则创建基本CSV文件
    if not os.path.exists(output_path):
        # 创建基本CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Dataset', 'Variable', 'Positions', 'Labels', 'Series', 'Question'])

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if content.startswith('<Sheet1>'):
        content = content.split('\n', 1)[1].replace('</Sheet1>', '')
    reader = csv.reader(content.splitlines())
    headers = next(reader)

    if 'Task' not in headers or 'Series' not in headers:
        raise ValueError(f"Missing required columns in {dataset_name} dataset")
    task_idx = headers.index('Task')
    series_idx = headers.index('Series')
    rows = list(reader)
    random.shuffle(rows)
    label_counter = {label: 0 for label in Task_TO_PROCESS}
    label_series = {label: [] for label in Task_TO_PROCESS}
    task_info = {}  # 存储Task信息
    processed_count = 0

    for row_idx, row in rows:
        if len(row) <= max(task_idx, series_idx):
            continue
        task_str = row[task_idx].strip()
        series_str = row[series_idx].strip()
        if not task_str or not series_str:
            continue
        try:
            tasks = ast.literal_eval(task_str)
            if not isinstance(tasks, list):
                continue
        except:
            continue
        try:
            if series_str.startswith('"') and series_str.endswith('"'):
                series_str = series_str[1:-1]
            series_data = normalize_series(ast.literal_eval(series_str))
            series_series = pd.Series(series_data)
            if series_series.isna().all():
                continue
        except:
            continue

        # 存储Task信息
        task_info[row_idx] = task_str

        if OUTPUT_DATASET_IMAGE:
            for label in tasks:
                if label not in Task_TO_PROCESS:
                    continue
                if label_counter[label] >= TSQA_SAMPLE_RAN_CNT:
                    continue
                label_dir = os.path.join(output_dir, get_label_dir_name(label))
                os.makedirs(label_dir, exist_ok=True)
                plt.figure(figsize=(10, 6))
                plt.plot(series_data, color='#1f77b4', linewidth=2)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(label_dir, f'{dataset_name}_{label}_{label_counter[label]}.png'), dpi=100)
                plt.close()
                label_series[label].append(series_data)
                label_counter[label] += 1
                processed_count += 1
    for label, series_list in label_series.items():
        if not series_list:
            continue
        long_series = []
        current_pos = 0
        positions = []
        for series in series_list:
            long_series.extend(normalize_series(series))
            positions.append((current_pos, current_pos + len(series) - 1))
            current_pos += len(series)
        if OUTPUT_DATASET_IMAGE:
            plt.figure(figsize=(20, 4))
            plt.plot(long_series, color='#1f77b4', linewidth=1.2)
            plt.title(f'{label} ({dataset_name})')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{get_label_dir_name(label)}_{dataset_name}.png")
            plt.savefig(output_path, dpi=150)
            plt.close()

    return f"Processed {dataset_name}: {processed_count} series"


def Streaming_time_series(label_data, dataset_name, task_info=None):
    input_path = DATASET_PATHS[dataset_name]
    dirname = os.path.dirname(input_path)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(dirname, f"stream-{filename}")
    os.makedirs(output_dir, exist_ok=True)

    # 使用传入的task_info，如果没有则为空字典
    if task_info is None:
        task_info = {}

    csv_path = os.path.join(output_dir, 'stream_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Dataset', 'Variable', 'Positions', 'Labels', 'Series', 'Question'])

        for i in tqdm(range(GEN_STREAM_CNT), desc=f"Generating {dataset_name} streaming samples"):
            # Generate one streaming data sequence
            long_series, labels_info, positions_info = generate_stream_series(label_data)
            # 注释掉二次归一化，因为在generate_stream_series中已经对每个series进行了归一化
            # long_series = normalize_series(long_series)
            # Check if sequence is empty
            if len(long_series) == 0:
                continue

            # 将Task内容拼接到Labels内容前
            modified_labels_info = []
            for j, label in enumerate(labels_info):
                # 从task_info中随机选择一个任务内容（因为我们无法确定具体对应关系）
                if task_info:
                    # 随机选择一个task内容
                    task_keys = list(task_info.keys())
                    if task_keys:
                        random_key = random.choice(task_keys)
                        task_content = task_info[random_key]
                        # 将Task内容拼接到Label前面
                        if isinstance(label, str):
                            modified_label = f"{task_content} {label}"
                        else:
                            modified_label = f"{task_content} {str(label)}"
                        modified_labels_info.append(modified_label)
                    else:
                        modified_labels_info.append(label)
                else:
                    modified_labels_info.append(label)

            # Plot and save sequence chart
            plot_and_save_series(long_series, i, output_dir)
            # 获取数据集变量信息
            variables = get_dataset_variables(dataset_name)
            # 对于单变量数据集，使用第一个变量名或默认值
            variable_list = [v.strip() for v in variables.split(',')] if variables else []
            variable_name = variable_list[0] if variable_list else dataset_name

            writer.writerow(
                [i + 1, dataset_name, variable_name, str(positions_info), str(modified_labels_info), str(long_series),
                 ''])
