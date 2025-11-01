#!/usr/bin/env python3
"""
数据加载器模块
负责从CSV文件加载数据并转换为JSON格式
"""

import json
import csv
import os
from typing import List
from data_structures import TemporalRecord


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_dir: str = "output"):
        self.data_dir = data_dir
    
    def load_csv_data(self) -> List[TemporalRecord]:
        """从CSV文件加载数据"""
        print("Loading CSV data...")
        records = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)
                print(f"  Processing file: {filename}")
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # 解析range字段
                            range_str = row['range'].strip('[]"')
                            range_parts = range_str.split(', ')
                            range_start = int(range_parts[0])
                            range_end = int(range_parts[1])
                            
                            record = TemporalRecord(
                                full_path=row['full_path'],
                                size=int(row['size']),
                                range_start=range_start,
                                range_end=range_end,
                                is_correct=row['is_correct'],
                                expected_normalized=row['expected_normalized'],
                                predicted_normalized=row['predicted_normalized'],
                                caption=row['caption'],
                                timestamp=row['timestamp']
                            )
                            records.append(record)
                        except (ValueError, KeyError) as e:
                            print(f"    Warning: Skipping invalid record - {e}")
                            continue
        
        print(f"Successfully loaded {len(records)} records")
        return records
    
    def convert_to_json(self, records: List[TemporalRecord]) -> None:
        """将数据转换为JSON格式并保存"""
        print("Converting data to JSON format...")
        
        json_data = []
        for record in records:
            json_record = {
                "full_path": record.full_path,
                "size": record.size,
                "range": [record.range_start, record.range_end],
                "is_correct": record.is_correct,
                "expected_normalized": record.expected_normalized,
                "predicted_normalized": record.predicted_normalized,
                "caption": record.caption,
                "timestamp": record.timestamp
            }
            json_data.append(json_record)
        
        # 保存为JSON文件
        json_filepath = os.path.join(self.data_dir, "temporal_data.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON data saved to: {json_filepath}")