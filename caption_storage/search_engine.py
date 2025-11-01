#!/usr/bin/env python3
"""
搜索引擎模块
负责执行基于缓存块的双维度检索算法
"""

from typing import Dict, List, Tuple, Set, Any
from data_structures import TemporalRecord, CacheBlock


class SearchEngine:
    """搜索引擎类"""
    
    def __init__(self, records: List[TemporalRecord], 
                 pattern_blocks: Dict[str, List[CacheBlock]],
                 time_range_index: Dict[Tuple[int, int], List[int]]):
        self.records = records
        self.pattern_blocks = pattern_blocks
        self.time_range_index = time_range_index
    
    def search_by_time_range(self, target_range: Tuple[int, int]) -> List[int]:
        """基于时间范围搜索记录ID"""
        target_start, target_end = target_range
        matching_ids = []
        
        for i, record in enumerate(self.records):
            # 检查时间范围重叠
            if (record.range_start <= target_end and record.range_end >= target_start):
                matching_ids.append(i)
        
        return matching_ids
    
    def search_by_pattern(self, patterns: List[str]) -> List[int]:
        """基于模式搜索记录ID"""
        matching_ids = set()
        
        for pattern in patterns:
            if pattern in self.pattern_blocks:
                for block in self.pattern_blocks[pattern]:
                    matching_ids.update(block.record_ids)
        
        return list(matching_ids)
    
    def search(self, time_range: Tuple[int, int] = None, 
               patterns: List[str] = None) -> List[Dict[str, Any]]:
        """执行综合搜索"""
        # 获取候选记录ID
        candidate_ids = set(range(len(self.records)))
        
        # 按时间范围过滤
        if time_range:
            time_matches = self.search_by_time_range(time_range)
            candidate_ids &= set(time_matches)
            print(f"  Time range matches: {len(time_matches)} records")
        
        # 按模式过滤
        if patterns:
            pattern_matches = self.search_by_pattern(patterns)
            candidate_ids &= set(pattern_matches)
            print(f"  Pattern matches: {len(pattern_matches)} records")
        
        # 构建结果
        results = []
        for record_id in candidate_ids:
            record = self.records[record_id]
            result = {
                "id": record_id,
                "range": [record.range_start, record.range_end],
                "pattern": record.predicted_normalized,
                "caption": record.caption,
                "is_correct": record.is_correct,
                "full_path": record.full_path,
                "expected_pattern": record.expected_normalized,
                "size": record.size,
                "timestamp": record.timestamp
            }
            results.append(result)
        
        # 按时间范围排序
        results.sort(key=lambda x: x["range"][0])
        
        print(f"Search completed, found {len(results)} matching records")
        return results
    
    def get_pattern_statistics(self) -> Dict[str, Dict[str, int]]:
        """获取模式统计信息"""
        pattern_stats = {}
        for pattern, blocks in self.pattern_blocks.items():
            total_records = sum(len(block.record_ids) for block in blocks)
            pattern_stats[pattern] = {
                'blocks': len(blocks),
                'records': total_records
            }
        return pattern_stats
    
    def get_time_range_coverage(self) -> Tuple[int, int]:
        """获取时间范围覆盖"""
        if not self.records:
            return (0, 0)
        
        min_start = min(record.range_start for record in self.records)
        max_end = max(record.range_end for record in self.records)
        return (min_start, max_end)
    
    def find_records_in_range(self, start: int, end: int, 
                             exact_match: bool = False) -> List[Dict[str, Any]]:
        """在指定范围内查找记录"""
        if exact_match:
            # 精确匹配时间范围
            matching_ids = self.time_range_index.get((start, end), [])
            results = []
            for record_id in matching_ids:
                record = self.records[record_id]
                results.append({
                    "id": record_id,
                    "range": [record.range_start, record.range_end],
                    "pattern": record.predicted_normalized,
                    "caption": record.caption
                })
        else:
            # 范围重叠匹配
            results = self.search(time_range=(start, end))
        
        return results
    
    def find_pattern_records(self, pattern: str) -> List[Dict[str, Any]]:
        """查找特定模式的所有记录"""
        return self.search(patterns=[pattern])