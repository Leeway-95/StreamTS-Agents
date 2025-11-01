#!/usr/bin/env python3
"""
智能查询解析器模块
使用语义相似度和实体识别进行智能查询理解
"""

import re
import jieba
import jieba.posseg as pseg
from typing import Optional, Tuple, List, Dict
import difflib


class SmartQueryParser:
    """基于智能语义理解的查询解析器类"""
    
    def __init__(self):
        self.known_patterns = [
            'upward trend',
            'downward trend', 
            'level shift outlier',
            'sudden spike outlier',
            'fixed seasonality',
            'shifting seasonality',
            'no temporal pattern',
            'obvious volatility'
        ]
        
        # 初始化jieba分词
        self._init_jieba()
    
    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加时间序列相关专业词汇
        jieba.add_word('时间范围', freq=1000, tag='n')
        jieba.add_word('异常值', freq=1000, tag='n')
        jieba.add_word('趋势', freq=1000, tag='n')

    def parse_query(self, query: str) -> Tuple[Optional[Tuple[int, int]], List[str]]:
        """解析自然语言查询，提取时间范围和模式关键词"""
        print(f"Query parsing: '{query}'")

        # 提取时间范围
        time_range = self._extract_time_range(query)

        # 智能提取模式
        detected_patterns = self._smart_pattern_detection(query)

        return time_range, detected_patterns

    def _extract_time_range(self, query: str) -> Optional[Tuple[int, int]]:
        """提取时间范围"""
        # 使用正则表达式匹配各种时间范围表达
        patterns = [
            r'\[(\d+),?\s*(\d+)\]',  # [数字, 数字]
            r'(\d+)\s*[-到至]\s*(\d+)',  # 数字-数字 或 数字到数字
            r'从\s*(\d+)\s*到\s*(\d+)',  # 从数字到数字
            r'(\d+)\s*和\s*(\d+)\s*之间'  # 数字和数字之间
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                print(f"Time range detected: [{start}, {end}]")
                return (start, end)

        return None
    
    def _smart_pattern_detection(self, query: str) -> List[str]:
        """智能模式检测"""
        # 分词处理
        tokens = list(jieba.cut(query))
        print(f"Tokenized: {tokens}")
        
        detected_patterns = []
        
        # 方法1: 直接匹配已知模式
        for pattern in self.known_patterns:
            if pattern.lower() in query.lower():
                detected_patterns.append(pattern)
        
        # 方法2: 语义相似度匹配
        if not detected_patterns:
            detected_patterns = self._semantic_similarity_matching(query, tokens)
        
        if detected_patterns:
            print(f"Patterns detected: {detected_patterns}")
        
        return detected_patterns
    
    def _semantic_similarity_matching(self, query: str, tokens: List[str]) -> List[str]:
        """基于语义相似度的模式匹配"""
        detected_patterns = []
        query_lower = query.lower()
        
        # 为每个已知模式计算相似度
        pattern_scores = {}
        
        for pattern in self.known_patterns:
            # 计算字符串相似度
            similarity = difflib.SequenceMatcher(None, query_lower, pattern.lower()).ratio()
            
            # 计算关键词重叠度
            pattern_words = set(pattern.lower().split())
            query_words = set(token.lower() for token in tokens)
            overlap = len(pattern_words & query_words) / len(pattern_words) if pattern_words else 0
            
            # 综合评分
            combined_score = similarity * 0.3 + overlap * 0.7
            pattern_scores[pattern] = combined_score
        
        # 选择评分最高且超过阈值的模式
        for pattern, score in pattern_scores.items():
            if score > 0.3:  # 相似度阈值
                detected_patterns.append(pattern)
        
        return detected_patterns
    
    def get_supported_patterns(self) -> List[str]:
        """获取支持的模式列表"""
        return self.known_patterns.copy()