import json
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *

logger = logging.getLogger(__name__)


class MemoryItem:
    """
    Memory item container for storing time series patterns and their related scores
    """
    __slots__ = ('series', 'position', 'r_score', 'i_score', 'hm')

    def __init__(self, series, position, r_score, i_score):
        """
        Initialize memory item

        Args:
            series: Time series data
            position: Position information
            r_score: Representativeness score
            i_score: Impact score
        """
        self.series = series
        self.position = position
        self.r_score = float(r_score)
        self.i_score = float(i_score)
        # 计算调和平均数作为综合分数
        self.hm = (2 * self.r_score * self.i_score) / (self.r_score + self.i_score + 1e-10)


class MemoryPool:
    """
    Memory pool for managing time series pattern memories
    """

    def __init__(self, max_size=DEFAULT_MEMORY_POOL_MAX_SIZE):
        """
        Initialize memory pool

        Args:
            max_size: Maximum capacity of memory pool
        """
        self.items = []
        self.max_size = max_size
        self.threshold = 0.0
        # 如果启用内存池保存且文件存在，则从文件加载
        if SAVE_MEMORY_POOL and os.path.exists(Memory_Pool_PATH):
            with open(Memory_Pool_PATH, 'r') as f:
                data = json.load(f)
                loaded_items = []
                for d in data:
                    item = MemoryItem(d['series'], d['position'], d['r_score'], d['i_score'])
                    loaded_items.append(item)
                # 按调和平均数排序
                loaded_items.sort(key=lambda x: x.hm, reverse=True)
                self.items = loaded_items[:max_size]

    def add_item(self, item):
        self.items.append(item)
        self.update_threshold()
        if SAVE_MEMORY_POOL:
            self.save_to_file()

    def update_threshold(self):
        if len(self.items) <= self.max_size:
            self.threshold = 0.0
            return
        # 按调和平均数排序
        sorted_items = sorted(self.items, key=lambda x: x.hm, reverse=True)
        if len(sorted_items) > self.max_size:
            self.threshold = sorted_items[self.max_size - 1].hm
            self.items = sorted_items[:self.max_size]

    def save_to_file(self):
        all_items = []
        if os.path.exists(Memory_Pool_PATH):
            with open(Memory_Pool_PATH, 'r') as f:
                all_items = json.load(f)
        # 将当前内存项添加到文件中
        for item in self.items:
            item_dict = {
                'series': item.series,
                'position': item.position,
                'r_score': item.r_score,
                'i_score': item.i_score
            }
            if item_dict not in all_items:
                all_items.append(item_dict)
        # 按调和平均数排序并限制数量
        all_items.sort(key=lambda x: (2 * float(x['r_score']) * float(x['i_score'])) / (
                float(x['r_score']) + float(x['i_score']) + 1e-10), reverse=True)
        all_items = all_items[:MEMORY_POOL_MAX_ITEMS]
        with open(Memory_Pool_PATH, 'w') as f:
            json.dump(all_items, f)

    def get_memory_patches(self):
        return [f"M_{idx}: {item.series} | Position: {item.position}" for idx, item in
                enumerate(self.items)]


def update_memory_pool(rep_series, position_info, r_scores, i_scores, memory_pool):
    if not rep_series or not position_info:
        return

    # 将代表性序列拆分为行
    rep_lines = rep_series.split('\n')
    valid_items = []

    # 获取最小长度，确保索引不会越界
    n = min(len(rep_lines), len(position_info), len(r_scores), len(i_scores))

    # 处理每个代表性序列
    for i in range(n):
        rep_line = rep_lines[i].strip()
        if not rep_line:
            continue
        try:
            pos_tuple = position_info[i]
            pos_str = str(pos_tuple)
            r_val = r_scores[i]
            i_val = float(i_scores[i])

            # 验证分数范围
            if r_val < 0 or r_val > 1 or i_val < 0 or i_val > 1:
                continue

            # 计算调和平均数
            score1 = 1 - r_val
            score2 = i_val
            hm = 2 * score1 * score2 / (score1 + score2) if (score1 + score2) != 0 else 0.0
            valid_items.append((hm, rep_line, pos_str, r_val, i_val))
        except Exception as e:
            logger.error(f"Error processing index {i}: {str(e)}")

    # 按调和平均数排序并选择前N个
    valid_items.sort(key=lambda x: x[0], reverse=True)
    top_items = valid_items[:MAX_TOP_HM_COUNT]

    # 创建新的内存项
    new_items = []
    for hm, rep_line, pos_str, r_val, i_val in top_items:
        try:
            new_item = MemoryItem(series=rep_line, position=pos_str, r_score=r_val, i_score=i_val)
            new_items.append(new_item)
        except Exception as e:
            logger.error(f"Error adding memory item: {str(e)}")

    # 将新项添加到内存池
    for item in new_items:
        memory_pool.add_item(item)