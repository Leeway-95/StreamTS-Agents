from matplotlib import pyplot as plt
from utils.common import *
from preprocess.variable_utils import get_dataset_variables
import csv
from tqdm import tqdm


def process_multi_dataset(dataset_path):
    # 根据 DATASET_PATHS 自动推断输出路径
    dirname = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
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

    # 读取数据集
    df = pd.read_csv(input_path)
    
    # 如果存在date列，则移除它
    if 'date' in df.columns:
        df = df.iloc[:, 1:]

    
    # 为每个列生成时间序列图表
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
    # 获取数据集变量信息
    variables = get_dataset_variables(dataset_name, input_path)
    # 将变量字符串分割为列表
    variable_list = [v.strip() for v in variables.split(',')] if variables else []
    
    for idx, col in enumerate(df.columns):
        # 获取对应的变量名，如果没有则使用列名
        variable_name = variable_list[idx] if idx < len(variable_list) else col
        output_data.append({
            'Index': idx + 1,
            'Dataset': dataset_name,
            'Variable': variable_name,
            'Positions': '[]',
            'Labels': '[]',
            'Series': str(normalize_series(df[col].tolist()))
        })
    
    # 保存到 CSV 文件
    pd.DataFrame(output_data).to_csv(output_path, index=False)
    
    return f"Processed {dataset_name}: {len(df.columns)} series"