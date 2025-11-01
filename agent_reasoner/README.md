# Caption-guided-CoT 模板系统

基于动态提示模板和Chain-of-Thought (CoT) 推理的数据处理系统优化实现。

## 📁 文件结构

```
agent_reasoner/
├── Caption-guided-CoT.txt          # 主模板文件
├── CoT/                            # CoT模板目录
│   ├── Reason.txt                  # 推理任务模板
│   ├── Understand.txt              # 理解任务模板
│   └── Forecast_Event.txt          # 事件预测模板
├── prompt_builder.py               # 提示构建器
├── playbook_manager_md.py          # Playbook管理器 (Markdown格式)
├── reasoner.py                     # 主系统接口
└── README.md                       # 本文档
```

## 🚀 使用方法

### 基本使用

```bash
# 激活环境
conda activate InfTS-LLM

# 运行推理任务
python reasoner.py -i "从时间范围[44000, 46000]中查找异常（level shift）"
python reasoner.py -i "How many upward trends in time range [1293495, 1294410]"
```
## 💡 核心特性

- **动态模板变量解析**: 支持 `{{ reasoning_task }}`、`{{ temporal_range }}`、`{{ caption }}` 等变量
- **Playbook缓存管理**: Markdown格式的经验知识自动积累
- **JSON输出格式**: 统一的结构化输出，包含 `playbook_updates`
- **命令行接口**: 支持直接输入查询进行测试

## 📋 输出格式

系统输出标准JSON格式：

```json
{
  "answer": "直接回答用户问题",
  "reasoning": "逐步推理过程",
  "playbook_updates": {
    "insights": ["新的洞察1", "新的洞察2"],
    "experiences": ["新的经验1", "新的经验2"], 
    "best_practices": ["最佳实践1", "最佳实践2"]
  }
}