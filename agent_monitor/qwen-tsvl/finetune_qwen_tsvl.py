#!/usr/bin/env python3
"""
Optimized Fine-tuning script for time series pattern classification model.
This script fixes the loss=0 and grad_norm=nan issues and adds comprehensive logging.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from typing import Dict, List, Tuple
import glob
from pathlib import Path
import logging
from datetime import datetime
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 设置环境变量支持多GPU并优化内存使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 使用三张GPU卡
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用多进程避免冲突
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 减少内存碎片

# 设置多进程启动方法为spawn以避免CUDA fork问题
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 如果已经设置过就忽略

# 增强的日志配置模块
LOG_DIR = "/data/liwei/log"
BACKUP_LOG_DIR = "./log"  # 备用日志目录

# 尝试创建主日志目录，如果失败则使用备用目录
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    # 测试写入权限
    test_file = os.path.join(LOG_DIR, "test_write.tmp")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print(f"使用主日志目录: {LOG_DIR}")
except (OSError, PermissionError) as e:
    print(f"无法使用主日志目录 {LOG_DIR}: {e}")
    LOG_DIR = BACKUP_LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"使用备用日志目录: {LOG_DIR}")

class EnhancedLogger:
    """增强的日志记录器类"""
    
    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        self.start_time = datetime.now()
        
        # 创建多个日志文件
        self.main_log_file = os.path.join(log_dir, f"finetune_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        self.error_log_file = os.path.join(log_dir, f"finetune_error_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        self.performance_log_file = os.path.join(log_dir, f"finetune_performance_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置主日志记录器
        self.main_logger = logging.getLogger('finetune_main')
        self.main_logger.setLevel(logging.DEBUG)
        
        # 配置错误日志记录器
        self.error_logger = logging.getLogger('finetune_error')
        self.error_logger.setLevel(logging.ERROR)
        
        # 配置性能日志记录器
        self.performance_logger = logging.getLogger('finetune_performance')
        self.performance_logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for logger in [self.main_logger, self.error_logger, self.performance_logger]:
            logger.handlers.clear()
        
        # 创建格式化器
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        performance_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s'
        )
        
        # 主日志处理器（文件 + 控制台）
        main_file_handler = logging.FileHandler(self.main_log_file, encoding='utf-8')
        main_file_handler.setFormatter(detailed_formatter)
        main_console_handler = logging.StreamHandler()
        main_console_handler.setFormatter(simple_formatter)
        
        self.main_logger.addHandler(main_file_handler)
        self.main_logger.addHandler(main_console_handler)
        
        # 错误日志处理器（仅文件）
        error_file_handler = logging.FileHandler(self.error_log_file, encoding='utf-8')
        error_file_handler.setFormatter(detailed_formatter)
        self.error_logger.addHandler(error_file_handler)
        
        # 性能日志处理器（仅文件）
        performance_file_handler = logging.FileHandler(self.performance_log_file, encoding='utf-8')
        performance_file_handler.setFormatter(performance_formatter)
        self.performance_logger.addHandler(performance_file_handler)
        
        # 记录初始化信息
        self.log_system_info()
    
    def log_system_info(self):
        """记录系统信息"""
        import platform
        
        self.main_logger.info("=" * 80)
        self.main_logger.info("FINETUNE MODEL 3 - 增强日志系统启动")
        self.main_logger.info("=" * 80)
        self.main_logger.info(f"启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info(f"Python版本: {platform.python_version()}")
        self.main_logger.info(f"操作系统: {platform.system()} {platform.release()}")
        
        if PSUTIL_AVAILABLE:
            self.main_logger.info(f"CPU核心数: {psutil.cpu_count()}")
            self.main_logger.info(f"内存总量: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        else:
            import os
            self.main_logger.info(f"CPU核心数: {os.cpu_count()}")
            self.main_logger.info("内存信息: psutil未安装，无法获取详细信息")
        
        if torch.cuda.is_available():
            self.main_logger.info(f"CUDA版本: {torch.version.cuda}")
            self.main_logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
            self.main_logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            self.main_logger.info("未检测到CUDA设备")
        
        self.main_logger.info(f"日志文件路径:")
        self.main_logger.info(f"  主日志: {self.main_log_file}")
        self.main_logger.info(f"  错误日志: {self.error_log_file}")
        self.main_logger.info(f"  性能日志: {self.performance_log_file}")
        self.main_logger.info("-" * 80)
    
    def info(self, message):
        """记录信息日志"""
        self.main_logger.info(message)
    
    def debug(self, message):
        """记录调试日志"""
        self.main_logger.debug(message)
    
    def warning(self, message):
        """记录警告日志"""
        self.main_logger.warning(message)
    
    def error(self, message, exc_info=None):
        """记录错误日志"""
        self.main_logger.error(message, exc_info=exc_info)
        self.error_logger.error(message, exc_info=exc_info)
    
    def performance(self, message):
        """记录性能日志"""
        self.performance_logger.info(message)
    
    def log_training_step(self, step, loss, lr, grad_norm=None):
        """记录训练步骤"""
        msg = f"Step {step} - Loss: {loss:.6f}, LR: {lr:.2e}"
        if grad_norm is not None:
            msg += f", Grad Norm: {grad_norm:.6f}"
        self.performance(msg)
    
    def log_epoch_summary(self, epoch, train_loss, val_loss=None, duration=None):
        """记录epoch摘要"""
        msg = f"Epoch {epoch} 完成 - 训练损失: {train_loss:.6f}"
        if val_loss is not None:
            msg += f", 验证损失: {val_loss:.6f}"
        if duration is not None:
            msg += f", 耗时: {duration:.2f}秒"
        self.performance(msg)
        self.info(msg)
    
    def log_dataset_info(self, dataset_name, total_samples, train_samples=None, val_samples=None):
        """记录数据集信息"""
        self.info(f"数据集 '{dataset_name}' 加载完成:")
        self.info(f"  总样本数: {total_samples}")
        if train_samples is not None:
            self.info(f"  训练样本: {train_samples}")
        if val_samples is not None:
            self.info(f"  验证样本: {val_samples}")
    
    def log_model_info(self, model_name, param_count=None):
        """记录模型信息"""
        self.info(f"模型加载: {model_name}")
        if param_count is not None:
            self.info(f"  参数数量: {param_count:,}")
    
    def finalize(self):
        """完成日志记录"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.main_logger.info("-" * 80)
        self.main_logger.info("训练完成")
        self.main_logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.main_logger.info(f"总耗时: {duration}")
        self.main_logger.info("=" * 80)

# 创建全局日志记录器实例
enhanced_logger = EnhancedLogger()
logger = enhanced_logger  # 保持向后兼容性

# Configuration
DATASET_ROOT = "/data/liwei/generator-5000"
input_model_path = "/data/liwei/model/Qwen3-VL-4B-Instruct"
output_model_path = "/data/liwei/model/Qwen3-VL-4B-Instruct-TS"
input_dataset = "/data/liwei/generator-5000/temporal-pattern-5000.json"
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
PATTERN_CATEGORIES = [
    "pattern_sudden_spike_outlier",
    "pattern_level_shift_outlier",
    "pattern_upward_trend",
    "pattern_downward_trend",
    "pattern_fixed_seasonality",
    "pattern_shifting_seasonality",
    "pattern_obvious_volatility",
    "no_temporal_pattern"
]

# Instruction template for time series analysis
INSTRUCTION = """You are a time series pattern recognition expert. Analyze the given time series image (blue line plot) and classify it into **exactly one dominant temporal pattern label** based on the following definitions and distinguishing visual cues.<\\s>"""
INPUT = """Focus on the overall shape, continuity, and repetition of the line.
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
Output strictly as: {"Label": "<one_of_the_above_labels>"}.<\\s>"""

class TimeSeriesDataset(Dataset):
    """优化的数据集类，修复标签处理问题"""

    def __init__(self, data_list: List[Dict], processor, max_length: int = 512):  # 增加max_length
        self.data_list = data_list
        self.processor = processor
        self.max_length = max_length
        enhanced_logger.log_dataset_info("TimeSeriesDataset", len(data_list))
        
        # 确保tokenizer有正确的特殊token
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            enhanced_logger.info("已设置 pad_token 为 eos_token")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            item = self.data_list[idx]

            # Load and process image
            image_path = item['image']
            if not os.path.exists(image_path):
                enhanced_logger.warning(f"图像文件未找到: {image_path}")
                # 返回一个默认的空白图像
                image = Image.new('RGB', (224, 224), color='white')
            else:
                image = Image.open(image_path).convert('RGB')

            # 准备输入文本（指令）- 添加图像占位符
            # Qwen3-VL需要在文本中包含<|vision_start|><|image_pad|><|vision_end|>来标识图像位置
            instruction_with_image = f"<|vision_start|><|image_pad|><|vision_end|>{item['instruction']}"

            # 准备目标标签 - 简化格式
            output_label = item['output']
            target_text = f'"{output_label}"'  # 简化的目标格式

            # 使用processor统一处理图像和文本 - 禁用截断以避免图像token被截断
            inputs = self.processor(
                text=instruction_with_image,
                images=image,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=False  # 禁用截断以避免图像token不匹配
            )
            
            # 确保image_grid_thw正确生成
            if 'image_grid_thw' not in inputs or inputs.get('image_grid_thw') is None:
                # 手动创建image_grid_thw：对于224x224图像，通常使用16x16的网格
                # 格式: [time_steps, height_grids, width_grids]
                # 对于静态图像: time_steps=1, 网格基于patch_size计算
                inputs['image_grid_thw'] = torch.tensor([[1, 16, 16]], dtype=torch.long)

            # 关键修复：确保标签长度与输入长度一致
            # 创建完整的输入-输出序列
            full_text = instruction_with_image + " " + target_text
            
            # 处理完整序列以获得正确的token化结果
            full_inputs = self.processor.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            # 处理指令部分以确定标签起始位置
            instruction_inputs = self.processor.tokenizer(
                instruction_with_image,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            # 创建标签：指令部分设为-100，输出部分保留token id
            labels = full_inputs['input_ids'].squeeze(0).clone()
            instruction_length = (instruction_inputs['attention_mask'].squeeze(0) == 1).sum().item()
            
            # 将指令部分的标签设为-100（不参与损失计算）
            labels[:instruction_length] = -100
            
            # 将padding部分也设为-100
            labels[full_inputs['attention_mask'].squeeze(0) == 0] = -100

            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if inputs['image_grid_thw'].dim() > 1 else inputs['image_grid_thw'],
                'labels': labels  # 现在标签长度与输入长度一致
            }

        except Exception as e:
            enhanced_logger.error(f"处理数据项 {idx} 时出错: {e}", exc_info=True)
            # 跳过有问题的数据项，返回None让DataLoader处理
            return None


class TimeSeriesDataProcessor:
    """优化的数据处理器"""

    def __init__(self, dataset_root: str = DATASET_ROOT):
        self.dataset_root = dataset_root
        enhanced_logger.info(f"数据处理器初始化，数据集根目录: {dataset_root}")

    def load_dataset_from_json(self, json_path: str = None) -> List[Dict]:
        """从JSON文件加载数据集"""
        if json_path is None:
            json_path = os.path.join(self.dataset_root, input_dataset)
        
        enhanced_logger.info(f"从 JSON 文件加载数据集: {json_path}")
        
        if not os.path.exists(json_path):
            enhanced_logger.error(f"数据集文件未找到: {json_path}")
            return []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            enhanced_logger.info(f"从 JSON 文件加载了 {len(data)} 个数据项")
            
            # 验证和处理数据
            processed_data = []
            for i, item in enumerate(data):
                try:
                    # 检查必需字段
                    if 'output' not in item or 'instruction' not in item:
                        enhanced_logger.warning(f"数据项 {i} 缺少必需字段，跳过处理")
                        continue
                    
                    # 处理图像路径
                    image_path = None
                    if 'images' in item and item['images']:
                        image_path = item['images'][0] if isinstance(item['images'], list) else item['images']
                    elif 'image' in item:
                        image_path = item['image']
                    
                    if image_path:
                        processed_item = {
                            "instruction": INSTRUCTION,
                            "input": INPUT,
                            "output": item['output'],
                            "image": image_path
                        }
                        processed_data.append(processed_item)
                        
                except Exception as e:
                    enhanced_logger.error(f"处理数据项 {i} 时出错: {e}", exc_info=True)
                    continue
            
            enhanced_logger.info(f"成功处理了 {len(processed_data)} 个数据项")
            return processed_data
            
        except Exception as e:
            enhanced_logger.error(f"加载数据集时出错: {e}", exc_info=True)
            return []

    def create_training_data_from_images(self) -> List[Dict]:
        """从图像目录创建训练数据"""
        enhanced_logger.info("从图像目录创建训练数据")
        training_data = []
        
        for pattern in PATTERN_CATEGORIES:
            pattern_dir = os.path.join(self.dataset_root, pattern)
            if not os.path.exists(pattern_dir):
                enhanced_logger.warning(f"模式目录未找到: {pattern_dir}")
                continue
            
            # 获取该模式的所有图像
            image_files = []
            for ext in IMAGE_EXTENSIONS:
                pattern_path = os.path.join(pattern_dir, f"*{ext}")
                image_files.extend(glob.glob(pattern_path))
            
            enhanced_logger.info(f"在模式 {pattern} 中找到 {len(image_files)} 个图像文件")
            
            # 为每个图像创建训练样本
            for image_path in image_files:
                label = pattern.replace('pattern_', '').replace('_', ' ')
                if pattern == 'no_pattern':
                    label = 'No Pattern'
                
                training_item = {
                    "instruction": INSTRUCTION,
                    "input": INPUT,
                    "output": label,
                    "image": image_path
                }
                training_data.append(training_item)
        
        enhanced_logger.info(f"从图像创建了 {len(training_data)} 个训练样本")
        return training_data


def collate_fn(batch):
    """自定义数据整理函数，确保批次数据一致性"""
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # 返回一个空的但格式正确的批次，而不是None
        return {
            'input_ids': torch.empty(0, dtype=torch.long),
            'attention_mask': torch.empty(0, dtype=torch.long),
            'pixel_values': torch.empty(0, dtype=torch.float32),
            'image_grid_thw': torch.empty(0, dtype=torch.long),
            'labels': torch.empty(0, dtype=torch.long)
        }
    
    # 获取批次中的最大长度
    max_input_length = max(item['input_ids'].size(0) for item in batch)
    max_label_length = max(item['labels'].size(0) for item in batch)
    
    # 确保所有序列长度一致
    input_ids = []
    attention_masks = []
    pixel_values = []
    image_grid_thws = []
    labels = []
    
    for item in batch:
        # 处理input_ids和attention_mask
        input_id = item['input_ids']
        attention_mask = item['attention_mask']
        
        # 如果长度不足，进行padding
        if input_id.size(0) < max_input_length:
            pad_length = max_input_length - input_id.size(0)
            input_id = torch.cat([input_id, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        # 处理labels
        label = item['labels']
        if label.size(0) < max_label_length:
            pad_length = max_label_length - label.size(0)
            label = torch.cat([label, torch.full((pad_length,), -100, dtype=torch.long)])
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        pixel_values.append(item['pixel_values'])
        
        # 确保image_grid_thw的维度正确 - 修复NoneType错误
        grid_thw = item['image_grid_thw']
        if grid_thw is not None:
            if grid_thw.dim() == 1:
                # 如果是1D，保持不变
                image_grid_thws.append(grid_thw)
            else:
                # 如果是2D，取第一行
                image_grid_thws.append(grid_thw[0] if grid_thw.size(0) > 0 else grid_thw.squeeze(0))
        else:
            # 如果为None，创建默认值
            image_grid_thws.append(torch.tensor([1, 16, 16], dtype=torch.long))
        
        labels.append(label)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'image_grid_thw': torch.stack(image_grid_thws),
        'labels': torch.stack(labels)
    }


class TimeSeriesTrainer:
    """优化的训练器类"""

    def __init__(self, model_name: str = input_model_path):
        self.model_name = model_name
        enhanced_logger.info(f"初始化训练器，使用模型: {model_name}")

        # 检测设备和多GPU配置
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            enhanced_logger.info(f"检测到 {gpu_count} 个GPU设备")
            for i in range(gpu_count):
                enhanced_logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.device = torch.device("cpu")
            enhanced_logger.info("使用 CPU")

        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                enhanced_logger.info("已清理GPU内存缓存")
            
            # 加载processor和model
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # 确保tokenizer有正确的特殊token - 修复token不匹配警告
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                enhanced_logger.info("已设置 pad_token 为 eos_token")
            
            # 显式设置特殊token ID以避免不匹配警告
            if hasattr(self.processor.tokenizer, 'eos_token_id') and self.processor.tokenizer.eos_token_id is not None:
                enhanced_logger.info(f"EOS token ID: {self.processor.tokenizer.eos_token_id}")
            if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
                enhanced_logger.info(f"PAD token ID: {self.processor.tokenizer.pad_token_id}")
            if hasattr(self.processor.tokenizer, 'bos_token_id') and self.processor.tokenizer.bos_token_id is not None:
                enhanced_logger.info(f"BOS token ID: {self.processor.tokenizer.bos_token_id}")
            
            # 使用内存优化方式加载模型
            enhanced_logger.info("开始加载模型...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  # 使用bfloat16减少内存使用
                low_cpu_mem_usage=True,
                device_map="auto",  # 自动分配到多个GPU
                trust_remote_code=True
            )
            
            # 同步模型配置与tokenizer的特殊token，解决token不匹配警告
            if hasattr(self.model.config, 'eos_token_id') and hasattr(self.processor.tokenizer, 'eos_token_id'):
                if self.model.config.eos_token_id != self.processor.tokenizer.eos_token_id:
                    enhanced_logger.info(f"更新模型config的eos_token_id: {self.model.config.eos_token_id} -> {self.processor.tokenizer.eos_token_id}")
                    self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
            
            if hasattr(self.model.config, 'pad_token_id') and hasattr(self.processor.tokenizer, 'pad_token_id'):
                if self.model.config.pad_token_id != self.processor.tokenizer.pad_token_id:
                    enhanced_logger.info(f"更新模型config的pad_token_id: {self.model.config.pad_token_id} -> {self.processor.tokenizer.pad_token_id}")
                    self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            
            if hasattr(self.model.config, 'bos_token_id') and hasattr(self.processor.tokenizer, 'bos_token_id'):
                if self.model.config.bos_token_id != self.processor.tokenizer.bos_token_id:
                    enhanced_logger.info(f"更新模型config的bos_token_id: {self.model.config.bos_token_id} -> {self.processor.tokenizer.bos_token_id}")
                    self.model.config.bos_token_id = self.processor.tokenizer.bos_token_id
            
            # 同步generation_config
            if hasattr(self.model, 'generation_config'):
                if hasattr(self.model.generation_config, 'eos_token_id') and hasattr(self.processor.tokenizer, 'eos_token_id'):
                    if self.model.generation_config.eos_token_id != self.processor.tokenizer.eos_token_id:
                        enhanced_logger.info(f"更新generation_config的eos_token_id: {self.model.generation_config.eos_token_id} -> {self.processor.tokenizer.eos_token_id}")
                        self.model.generation_config.eos_token_id = self.processor.tokenizer.eos_token_id
                
                if hasattr(self.model.generation_config, 'pad_token_id') and hasattr(self.processor.tokenizer, 'pad_token_id'):
                    if self.model.generation_config.pad_token_id != self.processor.tokenizer.pad_token_id:
                        enhanced_logger.info(f"更新generation_config的pad_token_id: {self.model.generation_config.pad_token_id} -> {self.processor.tokenizer.pad_token_id}")
                        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
                
                if hasattr(self.model.generation_config, 'bos_token_id') and hasattr(self.processor.tokenizer, 'bos_token_id'):
                    if self.model.generation_config.bos_token_id != self.processor.tokenizer.bos_token_id:
                        enhanced_logger.info(f"更新generation_config的bos_token_id: {self.model.generation_config.bos_token_id} -> {self.processor.tokenizer.bos_token_id}")
                        self.model.generation_config.bos_token_id = self.processor.tokenizer.bos_token_id
            
            # 模型已通过device_map自动分配到GPU，无需手动移动
            enhanced_logger.info("模型加载成功并自动分配到多GPU")
            
            # 显示模型设备分配信息
            if hasattr(self.model, 'hf_device_map'):
                enhanced_logger.info(f"模型设备分配: {self.model.hf_device_map}")
            
        except Exception as e:
            enhanced_logger.error(f"加载模型 {model_name} 时出错: {e}", exc_info=True)
            raise

    def prepare_datasets(self, data_list: List[Dict], train_ratio: float = 0.8):
        """准备训练和验证数据集"""
        enhanced_logger.log_dataset_info("训练数据集", len(data_list))
        
        # 打乱数据
        np.random.seed(42)  # 固定随机种子确保可重现
        np.random.shuffle(data_list)
        
        # 分割数据
        split_idx = int(len(data_list) * train_ratio)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
        
        enhanced_logger.log_dataset_info("数据集分割完成", len(data_list), len(train_data), len(val_data))
        
        # 创建数据集
        train_dataset = TimeSeriesDataset(train_data, self.processor)
        val_dataset = TimeSeriesDataset(val_data, self.processor)
        
        return train_dataset, val_dataset

    def train(self, data_list: List[Dict], output_dir: str = output_model_path):
        """训练模型"""
        enhanced_logger.info("开始模型训练")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据集
        train_dataset, val_dataset = self.prepare_datasets(data_list)
        
        # 训练参数 - 多GPU优化配置
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,  # 每个设备的批次大小保持较小
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,  # 增加梯度累积步数以补偿小批次
            learning_rate=5e-6,  # 更小的学习率确保稳定性
            max_grad_norm=1.0,  # 适度的梯度裁剪
            weight_decay=0.01,
            warmup_steps=20,  # 减少warmup步数
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=10,  # 更频繁的评估
            save_strategy="steps",
            save_steps=20,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=False,  # 禁用fp16
            bf16=True,   # 启用bf16以减少内存使用
            dataloader_num_workers=0,  # 避免多进程问题
            remove_unused_columns=False,
            logging_dir=os.path.join(LOG_DIR, "tensorboard"),
            seed=42,  # 固定随机种子
            dataloader_drop_last=True,  # 丢弃最后不完整的批次
            prediction_loss_only=True,  # 只计算预测损失，简化训练
            ddp_find_unused_parameters=False,  # 优化DDP性能
            dataloader_pin_memory=False,  # 减少内存使用
            gradient_checkpointing=True,  # 启用梯度检查点以减少内存
        )
        
        # 初始化训练器，使用自定义数据整理函数
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            data_collator=collate_fn,  # 使用自定义数据整理函数
        )
        
        # 开始训练
        enhanced_logger.info("训练开始...")
        try:
            trainer.train()
            enhanced_logger.info("训练成功完成")
            
            # 保存模型
            trainer.save_model(output_dir)
            self.processor.save_pretrained(output_dir)
            enhanced_logger.info(f"模型已保存到 {output_dir}")
            
        except Exception as e:
            enhanced_logger.error(f"训练失败: {e}", exc_info=True)
            raise


def main():
    """主函数"""
    enhanced_logger.info("=" * 60)
    enhanced_logger.info("时间序列模式分类微调 - 优化版本")
    enhanced_logger.info("=" * 60)
    
    try:
        # 初始化数据处理器
        processor = TimeSeriesDataProcessor()
        
        # 尝试从JSON文件加载数据
        enhanced_logger.info("正在加载数据集...")
        training_data = processor.load_dataset_from_json()
        
        # 如果JSON数据为空，从图像目录创建数据
        if not training_data:
            enhanced_logger.info("JSON数据集为空，从图像目录创建数据...")
            training_data = processor.create_training_data_from_images()
        
        if not training_data:
            enhanced_logger.error("没有可用的训练数据！")
            return
        
        # 显示数据集统计信息
        pattern_counts = {}
        for item in training_data:
            label = item['output']
            pattern_counts[label] = pattern_counts.get(label, 0) + 1
        
        enhanced_logger.info("数据集统计信息:")
        for label, count in pattern_counts.items():
            enhanced_logger.info(f"  {label}: {count} 个样本")
        
        # 初始化训练器
        enhanced_logger.info("正在初始化训练器...")
        trainer = TimeSeriesTrainer()
        
        # 开始训练
        enhanced_logger.info("开始微调过程...")
        trainer.train(training_data)
        
        enhanced_logger.info("微调成功完成！")
        enhanced_logger.finalize()  # 完成日志记录
        
    except Exception as e:
        enhanced_logger.error(f"主程序执行失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
