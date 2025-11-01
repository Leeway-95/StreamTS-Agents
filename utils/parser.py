#!/usr/bin/env python3
"""
Series数据解析工具模块
处理包含NaN值的Series数据解析问题
"""

import json
import ast
import logging
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def safe_parse_series(series_data, default_value=0.0):
    """
    安全解析Series数据，处理NaN值和各种异常情况
    
    Args:
        series_data: 要解析的Series数据（字符串、列表或其他类型）
        default_value: NaN值的替代值，默认为0.0
    
    Returns:
        list: 解析后的数值列表
    """
    # 处理None或空值情况
    if series_data is None:
        logger.debug("Series data is None, returning empty list")
        return []
    
    if pd.isna(series_data):
        logger.debug("Series data is NaN, returning empty list")
        return []
    
    # 如果已经是列表或元组，直接处理
    if isinstance(series_data, (list, tuple)):
        return _process_series_list(list(series_data), default_value)
    
    # 转换为字符串进行处理
    series_str = str(series_data).strip()
    if not series_str:
        logger.debug("Series data is empty string, returning empty list")
        return []
    
    # 尝试JSON解析
    try:
        parsed_data = json.loads(series_str)
        if isinstance(parsed_data, list):
            return _process_series_list(parsed_data, default_value)
        else:
            logger.warning(f"JSON parsed data is not a list: {type(parsed_data)}")
            return []
    except json.JSONDecodeError:
        logger.debug("JSON parsing failed, trying alternative methods")
    
    # 尝试使用ast.literal_eval解析
    try:
        parsed_data = ast.literal_eval(series_str)
        if isinstance(parsed_data, (list, tuple)):
            return _process_series_list(list(parsed_data), default_value)
        else:
            logger.warning(f"AST parsed data is not a list: {type(parsed_data)}")
            return []
    except (ValueError, SyntaxError):
        logger.debug("AST parsing failed, trying NaN replacement")
    
    # 尝试替换NaN值后再解析
    try:
        # 替换各种形式的NaN
        cleaned_str = _clean_nan_values(series_str, default_value)
        parsed_data = json.loads(cleaned_str)
        if isinstance(parsed_data, list):
            return _process_series_list(parsed_data, default_value)
    except json.JSONDecodeError:
        logger.debug("NaN replacement and JSON parsing failed")
    
    # 尝试手动提取数值
    try:
        numbers = _extract_numbers_from_string(series_str, default_value)
        if numbers:
            return numbers
    except Exception as e:
        logger.debug(f"Manual number extraction failed: {e}")
    
    # 所有方法都失败，返回空列表
    logger.warning(f"Unable to parse Series field: {series_str[:50]}...")
    return []

def _process_series_list(data_list, default_value=0.0):
    """
    处理解析后的列表，替换NaN值
    
    Args:
        data_list: 要处理的数据列表
        default_value: NaN值的替代值
    
    Returns:
        list: 处理后的数值列表
    """
    processed_list = []
    for item in data_list:
        if item is None or pd.isna(item):
            processed_list.append(default_value)
        elif isinstance(item, str):
            if item.lower() in ['nan', 'null', 'none', '']:
                processed_list.append(default_value)
            else:
                try:
                    processed_list.append(float(item))
                except ValueError:
                    logger.warning(f"Cannot convert '{item}' to float, using default value")
                    processed_list.append(default_value)
        else:
            try:
                # 尝试转换为浮点数
                processed_list.append(float(item))
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert {item} (type: {type(item)}) to float, using default value")
                processed_list.append(default_value)
    
    return processed_list

def _clean_nan_values(series_str, default_value=0.0):
    """
    清理字符串中的NaN值
    
    Args:
        series_str: 要清理的字符串
        default_value: NaN值的替代值
    
    Returns:
        str: 清理后的字符串
    """
    # 替换各种形式的NaN
    cleaned_str = series_str
    
    # 使用正则表达式替换NaN值
    nan_patterns = [
        r'\bnan\b',      # nan
        r'\bNaN\b',      # NaN
        r'\bNAN\b',      # NAN
        r'\bnull\b',     # null
        r'\bNull\b',     # Null
        r'\bNULL\b',     # NULL
        r'\bNone\b',     # None
    ]
    
    for pattern in nan_patterns:
        cleaned_str = re.sub(pattern, str(default_value), cleaned_str, flags=re.IGNORECASE)
    
    return cleaned_str

def _extract_numbers_from_string(series_str, default_value=0.0):
    """
    从字符串中手动提取数值
    
    Args:
        series_str: 包含数值的字符串
        default_value: 无效数值的替代值
    
    Returns:
        list: 提取的数值列表
    """
    # 检查是否看起来像一个数组
    if not (series_str.strip().startswith('[') and series_str.strip().endswith(']')):
        return []
    
    # 移除方括号
    content = series_str.strip()[1:-1]
    
    # 按逗号分割
    items = content.split(',')
    
    numbers = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        # 检查是否是NaN值
        if item.lower() in ['nan', 'null', 'none']:
            numbers.append(default_value)
            continue
        
        # 尝试转换为数值
        try:
            numbers.append(float(item))
        except ValueError:
            logger.warning(f"Cannot parse item '{item}' as number, using default value")
            numbers.append(default_value)
    
    return numbers

def validate_series_data(series_data, expected_length=None):
    """
    验证Series数据的有效性
    
    Args:
        series_data: 要验证的Series数据
        expected_length: 期望的数据长度（可选）
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(series_data, list):
        return False, f"Series data is not a list: {type(series_data)}"
    
    if not series_data:
        return False, "Series data is empty"
    
    if expected_length is not None and len(series_data) != expected_length:
        return False, f"Series length mismatch: expected {expected_length}, got {len(series_data)}"
    
    # 检查是否包含有效数值
    valid_count = 0
    for item in series_data:
        if isinstance(item, (int, float)) and not pd.isna(item):
            valid_count += 1
    
    if valid_count == 0:
        return False, "Series data contains no valid numbers"
    
    return True, "Series data is valid"