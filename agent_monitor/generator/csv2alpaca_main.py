"""
CSV to Alpaca 格式转换主执行文件
包含配置参数和主要执行逻辑
"""

import json
import subprocess
import argparse
from data_converter import create_alpaca_format
from data_synthesizer import generate_synthetic_data, generate_no_pattern_data
from image_generator import generate_all_pattern_images
from file_operations import save_alpaca_json

# 配置参数
cnt_synthetic_pattern_sample = 3800
cnt_synthetic_no_pattern_sample = 1000
cnt_original_pattern_sample = 200

# cnt_synthetic_pattern_sample = 700
# cnt_synthetic_no_pattern_sample = 100
# cnt_original_pattern_sample = 200

# 根目录路径常量
ROOT_PATH = "/data/liwei/datase/"
DATASET_DIR = "dataset"

# 指令文本
instruction = (
    "You are a time series pattern recognition expert. Analyze the given time series image (blue line plot) and classify it into **exactly one dominant temporal pattern label** based on the following definitions and distinguishing visual cues.<\\s>")

# 所有可用的模式类型
ALL_PATTERNS = [
    "Sudden Spike Outlier",
    "Level Shift Outlier",
    "Fixed Seasonality",
    "Shifting Seasonality",
    "Upward Trend",
    "Downward Trend",
    "Obvious Volatility",
    "No Temporal Pattern"
]

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='生成时间序列数据集，支持指定模式类型和策略')
    parser.add_argument(
        '--patterns',
        nargs='*',
        choices=ALL_PATTERNS,
        default=ALL_PATTERNS,
        help='指定要生成的模式类型，可选择多个。如果不指定，则生成所有模式。'
    )
    parser.add_argument(
        '--synthetic-samples',
        type=int,
        default=cnt_synthetic_pattern_sample,
        help=f'合成数据样本数量 (默认: {cnt_synthetic_pattern_sample})'
    )
    parser.add_argument(
        '--no-pattern-samples',
        type=int,
        default=cnt_synthetic_no_pattern_sample,
        help=f'无模式数据样本数量 (默认: {cnt_synthetic_no_pattern_sample})'
    )
    parser.add_argument(
        '--strategy',
        choices=['depth', 'breadth'],
        default='depth',
        help='数据生成策略：depth=每张图片只有一种模式，breadth=每张图片可叠加多种模式 (默认: depth)'
    )
    
    return parser.parse_args()

def main():
    """
    主函数：执行完整的转换流程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("开始读取TSQA.csv文件并生成数据...")
    print(f"指定生成的模式类型: {args.patterns}")
    print(f"合成数据样本数量: {args.synthetic_samples}")
    print(f"无模式数据样本数量: {args.no_pattern_samples}")
    print(f"数据生成策略: {args.strategy}")

    # 创建Alpaca格式数据，不生成可视化图片
    alpaca_data = create_alpaca_format(instruction, generate_images=False)

    if not alpaca_data:
        print("没有成功读取到数据")
        return

    print(f"成功读取 {len(alpaca_data)} 条原始数据。开始生成额外的合成数据...")
    
    # 计算原始数据的全局位置，用于后续合成数据的range计算
    global_position = 0
    if alpaca_data:
        # 获取原始数据中最后一个range的结束位置
        last_range = alpaca_data[-1].get('range', [0, 0])
        global_position = last_range[1]
    
    # 根据指定的模式和策略生成合成数据
    synthetic_data, global_position = generate_synthetic_data(
        alpaca_data, args.synthetic_samples, instruction,
        selected_patterns=args.patterns, strategy=args.strategy,
        global_position=global_position
    )

    # 将合成数据添加到原始数据后面
    all_data = alpaca_data + synthetic_data

    # 如果指定的模式包含"No Temporal Pattern"，则生成对应数据
    if "No Temporal Pattern" in args.patterns:
        no_pattern_data, global_position = generate_no_pattern_data(
            args.no_pattern_samples, instruction, global_position=global_position
        )
        all_data.extend(no_pattern_data)
    
    # 为其他模式数据生成按模式分类的图片
    all_data, pattern_stats = generate_all_pattern_images(all_data, DATASET_DIR, ROOT_PATH, selected_patterns=args.patterns)

    # 显示第一条数据作为示例
    if all_data:
        print("\n第一条数据示例:")
        print(json.dumps(all_data[0], ensure_ascii=False, indent=2))

    # 保存为JSON文件到dataset目录
    total_samples = len(all_data)
    filename = f'dataset/temporal-pattern.json'
    print(f"\n正在保存到 {filename}...")
    success = save_alpaca_json(all_data, filename)

    if success:
        print("转换完成！")
        print("生成了按模式分类的全量图片到 dataset/ 目录:")
        for folder_name, count in pattern_stats.items():
            if count > 0:
                print(f"  dataset/{folder_name}: {count} 张")
        print(f"总共保存了 {len(all_data)} 条数据到 {filename}")
        print(f"所有图片路径已添加根目录前缀: {ROOT_PATH}")
        print(f"生成的模式类型: {args.patterns}")
    else:
        print("保存失败！")


if __name__ == "__main__":
    main()