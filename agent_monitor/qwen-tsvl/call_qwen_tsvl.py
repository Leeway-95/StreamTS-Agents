#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改后的call_model_demo.py
功能：遍历dataset目录下所有图片，使用模型进行推理，输出CSV结果
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import csv
from datetime import datetime
from pathlib import Path
import re
from tqdm import tqdm

model_path = "/data/liwei/model/Qwen3-TSVL-4B"
dataset_path = Path("/data/liwei/dataset")

def extract_pattern_from_path(path):
    """从目录路径中提取模式名称"""
    path_parts = Path(path).parts
    for part in path_parts:
        if part.startswith('pattern_'):
            return part.replace('pattern_', '').replace('_', ' ')
        elif part == 'no_pattern':
            return 'No Pattern'
    return None


def normalize_pattern_name(pattern):
    """标准化模式名称用于比较"""
    if not pattern:
        return ""
    return pattern.strip().lower()


def extract_json_from_response(response_text):
    """从模型响应中提取JSON格式的标签和描述"""
    try:
        # 尝试直接解析JSON
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            if 'Label' in parsed:
                label = parsed['Label']
                caption = parsed.get('Caption', '')  # 获取Caption，如果不存在则为空字符串
                return label, caption
    except:
        pass

    # 如果JSON解析失败，尝试从文本中提取模式
    patterns = [
        'Sudden Spike Outlier', 'Level Shift Outlier',
        'Upward Trend', 'Downward Trend',
        'Fixed Seasonality', 'Shifting Seasonality',
        'Obvious Volatility', 'No Temporal Pattern'
    ]

    response_lower = response_text.lower()
    for pattern in patterns:
        if pattern.lower() in response_lower:
            return pattern, ''  # 返回模式和空描述

    return "No Pattern", ''  # 返回默认值和空描述


def load_model_and_processor():
    """加载模型和处理器"""
    print("正在加载Qwen3-TSVL-4B模型...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_path)

    print("模型加载完成！")
    return model, processor


def process_single_image(model, processor, image_path, prompt_text):
    """处理单张图片"""
    try:
        # 验证图片文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return "Error"
        
        # 构建输入消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

        # 准备推理输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息
        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception as e:
            print(f"处理视觉信息时出错 {image_path}: {str(e)}")
            return "Error"

        # 处理输入张量
        try:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"处理输入张量时出错 {image_path}: {str(e)}")
            return "Error"

        # 将输入移动到模型设备
        try:
            inputs = inputs.to(model.device)
        except Exception as e:
            print(f"移动到设备时出错 {image_path}: {str(e)}")
            return "Error"

        # 执行推理
        try:
            with torch.no_grad():  # 添加no_grad以节省内存
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
        except Exception as e:
            print(f"模型推理时出错 {image_path}: {str(e)}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "Error"

        # 解码输出
        try:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_text[0]
        except Exception as e:
            print(f"解码输出时出错 {image_path}: {str(e)}")
            return "Error"

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Error"


def sort_files_naturally(files):
    """按自然顺序排序文件"""

    def natural_sort_key(filename):
        # 将文件名中的数字转换为整数，实现自然排序
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', str(filename))]

    return sorted(files, key=natural_sort_key)


def write_csv_results(csv_filename, results_data):
    """将结果写入CSV文件"""
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['full_path', 'size', 'range', 'is_correct', 'expected_normalized',
                      'predicted_normalized', 'caption', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results_data:
            writer.writerow(row)


def load_range_mapping():
    """加载temporal-pattern.json文件并创建图片路径到range的映射"""
    range_mapping = {}
    try:
        with open('dataset/temporal-pattern.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'images' in item and 'range' in item:
                    for image_path in item['images']:
                        # 提取完整路径和文件名作为键，支持多种匹配方式
                        full_path = str(image_path)
                        filename = Path(image_path).name
                        
                        # 同时存储完整路径和文件名的映射
                        range_mapping[full_path] = item['range']
                        range_mapping[filename] = item['range']
                        
                        # 处理各种可能的路径变化
                        # 1. 处理datase -> dataset的拼写修正
                        if '/datase/' in full_path:
                            corrected_path = full_path.replace('/datase/', '/dataset/')
                            range_mapping[corrected_path] = item['range']
                            # 提取相对路径
                            if '/dataset/' in corrected_path:
                                relative_path = corrected_path.split('/dataset/')[-1]
                                range_mapping[relative_path] = item['range']
                        
                        # 2. 处理dataset-5000的情况
                        if '/dataset-5000/' in full_path:
                            relative_path = full_path.split('/dataset-5000/')[-1]
                            range_mapping[relative_path] = item['range']
                        
                        # 3. 处理dataset的情况
                        if '/dataset/' in full_path:
                            relative_path = full_path.split('/dataset/')[-1]
                            range_mapping[relative_path] = item['range']
                        
                        # 4. 提取pattern目录和文件名的组合
                        path_parts = Path(full_path).parts
                        for i, part in enumerate(path_parts):
                            if part.startswith('pattern_'):
                                pattern_filename = f"{part}/{filename}"
                                range_mapping[pattern_filename] = item['range']
                                break
                                
    except Exception as e:
        print(f"加载temporal-pattern.json时出错: {e}")
    return range_mapping


def main():
    """主函数"""
    print("开始批量图片推理任务...")

    # 加载range映射
    range_mapping = load_range_mapping()
    print(f"加载了 {len(range_mapping)} 个图片的range信息")

    # 加载模型
    model, processor = load_model_and_processor()

    # 读取提示词
    prompt_text = """You are a time series pattern recognition expert. 
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

    # 统计变量
    results = {}
    total_files = 0
    correct_predictions = 0

    # CSV数据列表
    csv_data = []

    if not dataset_path.exists():
        print("dataset目录不存在")
        return

    # 获取所有子目录
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    # 计算总文件数用于进度条
    all_image_files = []
    for subdir in subdirs:
        image_files = list(subdir.glob("*.png"))
        # 按自然顺序排序图片文件
        sorted_image_files = sort_files_naturally(image_files)
        for img_file in sorted_image_files:
            all_image_files.append((subdir, img_file))

    total_files = len(all_image_files)
    if total_files == 0:
        print("没有找到任何图片文件")
        return

    print(f"找到 {total_files} 个图片文件，开始处理...")

    # 使用进度条处理所有图片
    with tqdm(total=total_files, desc="处理进度", unit="张") as pbar:
        for subdir, image_file in all_image_files:
            dir_name = subdir.name
            expected_pattern = extract_pattern_from_path(str(subdir))

            if not expected_pattern:
                pbar.update(1)
                continue

            # 初始化该目录的统计（如果不存在）
            if dir_name not in results:
                results[dir_name] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'accuracy': 0.0
                }

            # 进行推理
            response = process_single_image(model, processor, image_file, prompt_text)

            # 提取预测结果和描述
            predicted_pattern, caption = extract_json_from_response(response)

            # 比较结果
            expected_normalized = normalize_pattern_name(expected_pattern)
            predicted_normalized = normalize_pattern_name(predicted_pattern)

            is_correct = expected_normalized == predicted_normalized

            # 更新统计
            results[dir_name]['total'] += 1
            if is_correct:
                correct_predictions += 1
                results[dir_name]['correct'] += 1
            else:
                results[dir_name]['incorrect'] += 1

            # 获取range信息，尝试多种匹配方式
            filename = image_file.name
            full_path = str(image_file)
            range_info = []
            
            # 尝试多种匹配方式
            matching_keys = [
                full_path,  # 完整路径
                filename,   # 文件名
            ]
            
            # 添加相对路径匹配
            if '/dataset/' in full_path:
                relative_path = full_path.split('/dataset/')[-1]
                matching_keys.append(relative_path)
            
            if '/dataset-5000/' in full_path:
                relative_path = full_path.split('/dataset-5000/')[-1]
                matching_keys.append(relative_path)
            
            # 添加pattern目录+文件名的匹配
            path_parts = Path(full_path).parts
            for i, part in enumerate(path_parts):
                if part.startswith('pattern_'):
                    pattern_filename = f"{part}/{filename}"
                    matching_keys.append(pattern_filename)
                    break
            
            # 尝试所有匹配键
            for key in matching_keys:
                if key in range_mapping:
                    range_info = range_mapping[key]
                    break
            
            range_str = f"[{range_info[0]}, {range_info[1]}]" if len(range_info) == 2 else ""
            # 计算size（range的长度）
            size_value = range_info[1] - range_info[0] + 1 if len(range_info) == 2 else ""

            # 添加到CSV数据
            csv_data.append({
                'full_path': str(image_file),
                'size': size_value,
                'range': range_str,
                'is_correct': '正确' if is_correct else '错误',
                'expected_normalized': expected_normalized,
                'predicted_normalized': predicted_normalized,
                'caption': caption,
                'timestamp': datetime.now().isoformat()
            })

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix_str(f"当前: {dir_name}")

    # 计算各目录正确率
    for dir_name, stats in results.items():
        stats['accuracy'] = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0

    # 输出总体统计
    overall_accuracy = (correct_predictions / total_files * 100) if total_files > 0 else 0

    print("=" * 60)
    print("原始模型任务完成总结")
    print("=" * 60)
    print(f"总文件数: {total_files}")
    print(f"正确预测: {correct_predictions}")
    print(f"错误预测: {total_files - correct_predictions}")
    print(f"总体正确率: {overall_accuracy:.2f}%")

    # 保存CSV结果
    csv_filename = f"/data/liwei/output/inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv_results(csv_filename, csv_data)
    print(f"详细结果已保存到: {csv_filename}")

    # 保存统计结果到JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': total_files,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': total_files - correct_predictions,
        'overall_accuracy': overall_accuracy,
        'directory_results': results
    }
    json_filename = f"/data/liwei/log/inference_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"统计结果已保存到: {json_filename}")


if __name__ == "__main__":
    main()
