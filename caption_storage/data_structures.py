#!/usr/bin/env python3
"""
时间序列数据结构定义模块
定义系统中使用的核心数据结构
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TemporalRecord:
    """时间序列记录数据结构"""
    full_path: str
    size: int
    range_start: int
    range_end: int
    is_correct: str
    expected_normalized: str
    predicted_normalized: str
    caption: str
    timestamp: str


@dataclass
class CacheBlock:
    """缓存块数据结构"""
    pattern: str
    ranges: List[Tuple[int, int]]  # (start, end) tuples
    captions: List[str]
    record_ids: List[int]  # 对应原始记录的ID