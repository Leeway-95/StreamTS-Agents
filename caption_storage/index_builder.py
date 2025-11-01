#!/usr/bin/env python3
"""
索引构建器模块
负责构建和管理双维度索引结构
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
from data_structures import TemporalRecord, CacheBlock


class IndexBuilder:
    """索引构建器类"""
    
    def __init__(self, index_dir: str = "index_cache"):
        self.index_dir = index_dir
        self.pattern_blocks: Dict[str, List[CacheBlock]] = defaultdict(list)
        self.time_range_index: Dict[Tuple[int, int], List[int]] = {}
        
        # 创建索引目录
        os.makedirs(index_dir, exist_ok=True)
    
    def build_pattern_blocks(self, records: List[TemporalRecord]) -> None:
        """构建基于模式的缓存块索引"""
        print("Building pattern cache block index...")
        
        # 按模式分组记录
        pattern_groups = defaultdict(list)
        for i, record in enumerate(records):
            pattern = record.predicted_normalized
            pattern_groups[pattern].append((i, record))
        
        # 为每个模式创建缓存块
        for pattern, records_with_ids in pattern_groups.items():
            # 按时间范围排序
            records_with_ids.sort(key=lambda x: x[1].range_start)
            
            # 创建缓存块（每个块最多包含50个记录以优化性能）
            block_size = 50
            for i in range(0, len(records_with_ids), block_size):
                block_records = records_with_ids[i:i + block_size]
                
                cache_block = CacheBlock(
                    pattern=pattern,
                    ranges=[(r.range_start, r.range_end) for _, r in block_records],
                    captions=[r.caption for _, r in block_records],
                    record_ids=[rid for rid, _ in block_records]
                )
                
                self.pattern_blocks[pattern].append(cache_block)
        
        print(f"Created {sum(len(blocks) for blocks in self.pattern_blocks.values())} cache blocks")
        print(f"   Covering {len(self.pattern_blocks)} pattern types")
    
    def build_time_range_index(self, records: List[TemporalRecord]) -> None:
        """构建时间范围索引"""
        print("Building time range index...")
        
        for i, record in enumerate(records):
            range_key = (record.range_start, record.range_end)
            if range_key not in self.time_range_index:
                self.time_range_index[range_key] = []
            self.time_range_index[range_key].append(i)
        
        print(f"Created {len(self.time_range_index)} time range index entries")
    
    def save_index(self, records: List[TemporalRecord]) -> None:
        """保存索引到文件"""
        print("Saving index files...")
        
        # 保存模式块索引
        pattern_blocks_file = os.path.join(self.index_dir, "pattern_blocks.pkl")
        with open(pattern_blocks_file, 'wb') as f:
            pickle.dump(dict(self.pattern_blocks), f)
        
        # 保存时间范围索引
        time_range_file = os.path.join(self.index_dir, "time_range_index.pkl")
        with open(time_range_file, 'wb') as f:
            pickle.dump(self.time_range_index, f)
        
        # 保存记录数据
        records_file = os.path.join(self.index_dir, "records.pkl")
        with open(records_file, 'wb') as f:
            pickle.dump(records, f)
        
        print(f"Index saved to {self.index_dir} directory")
    
    def load_index(self) -> Tuple[bool, List[TemporalRecord]]:
        """从文件加载索引"""
        try:
            print("Loading existing index...")
            
            # 加载模式块索引
            pattern_blocks_file = os.path.join(self.index_dir, "pattern_blocks.pkl")
            with open(pattern_blocks_file, 'rb') as f:
                self.pattern_blocks = defaultdict(list, pickle.load(f))
            
            # 加载时间范围索引
            time_range_file = os.path.join(self.index_dir, "time_range_index.pkl")
            with open(time_range_file, 'rb') as f:
                self.time_range_index = pickle.load(f)
            
            # 加载记录数据
            records_file = os.path.join(self.index_dir, "records.pkl")
            with open(records_file, 'rb') as f:
                records = pickle.load(f)
            
            print(f"Successfully loaded index ({len(records)} records)")
            return True, records
            
        except FileNotFoundError:
            print("No existing index found, will rebuild")
            return False, []
    
    def get_pattern_blocks(self) -> Dict[str, List[CacheBlock]]:
        """获取模式块索引"""
        return dict(self.pattern_blocks)
    
    def get_time_range_index(self) -> Dict[Tuple[int, int], List[int]]:
        """获取时间范围索引"""
        return self.time_range_index