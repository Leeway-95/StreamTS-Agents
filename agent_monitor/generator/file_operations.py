"""
文件操作模块
包含JSON文件保存和其他文件操作相关的功能
"""

import json


def save_alpaca_json(data, filename):
    """
    将Alpaca格式的数据保存为JSON文件
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"成功保存 {len(data)} 条数据到 {filename}")
        return True
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
        return False