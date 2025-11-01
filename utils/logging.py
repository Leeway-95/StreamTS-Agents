"""
日志配置模块
提供项目统一的日志配置和工具类
"""

import logging
import sys
from typing import TextIO


class Tee:
    """
    同时输出到多个流的工具类
    可以同时向标准输出和文件写入内容
    """
    
    def __init__(self, *files: TextIO):
        """
        初始化Tee对象
        
        Args:
            *files: 要写入的文件对象列表
        """
        self.files = files
    
    def write(self, obj: str) -> None:
        """
        写入内容到所有文件流
        
        Args:
            obj: 要写入的字符串内容
        """
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保内容立即写入
    
    def flush(self) -> None:
        """刷新所有文件流"""
        for f in self.files:
            f.flush()


def get_project_logger(name: str) -> logging.Logger:
    """
    获取项目标准日志记录器
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        
    Returns:
        配置好的日志记录器实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 设置日志级别
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(console_handler)
        
        # 防止日志向上传播到根日志记录器
        logger.propagate = False
    
    return logger


def setup_file_logger(logger: logging.Logger, log_file_path: str) -> None:
    """
    为日志记录器添加文件处理器
    
    Args:
        logger: 日志记录器实例
        log_file_path: 日志文件路径
    """
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 添加文件处理器
    logger.addHandler(file_handler)


def configure_root_logger(level: int = logging.INFO) -> None:
    """
    配置根日志记录器
    
    Args:
        level: 日志级别，默认为 INFO
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )