# Caption Block - 时间序列模式索引和检索系统

高效的JSON文件索引和检索系统，专门用于时间序列模式分析数据。

## 功能特性

- **双维度索引**：时间范围 + 时间模式的高效索引结构
- **智能查询**：支持中英文自然语言查询，自动识别时间范围和模式关键词
- **高性能**：毫秒级查询响应时间，基于缓存块设计
- **模块化架构**：清晰的6层模块分离，便于维护和扩展

## 快速开始

### 安装依赖
```bash
pip install jieba
```

### 基本使用
```bash
# 单次查询
python caption_block.py -i "从时间范围[44000, 46000]中查找异常（level shift）"

# 交互模式
python caption_block.py --interactive

# 查看统计信息
python caption_block.py --stats
```

### 支持的查询示例
- `从时间范围[44000, 46000]中查找异常（level shift）`
- `寻找上升趋势`
- `upward trend`
- `时间范围[45000, 46000]`
- `突刺异常`

## 系统架构

```
├── caption_block.py        # 主系统接口
├── data_structures.py      # 数据结构定义
├── data_loader.py          # 数据加载器
├── index_builder.py        # 索引构建器
├── smart_query_parser.py   # 智能查询解析器
└── search_engine.py        # 搜索引擎
```

## 支持的时间模式

- `upward trend` - 上升趋势
- `downward trend` - 下降趋势
- `level shift outlier` - 水平移位异常
- `sudden spike outlier` - 突刺异常
- `no temporal pattern` - 无时间模式
- `fixed seasonality` - 固定季节性
- `obvious volatility` - 明显波动性

## 性能指标

- 查询响应时间：< 2ms
- 支持数据量：906条记录
- 索引构建时间：< 5秒
- 内存使用：缓存块优化设计