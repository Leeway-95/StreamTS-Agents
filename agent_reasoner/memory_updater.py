"""
内存更新模块
用于更新和管理内存池中的时间序列数据
"""

def update_memory_pool(data, memory_pool):
    """
    更新内存池中的数据
    
    Args:
        data: 新的时间序列数据
        memory_pool: 当前的内存池
    
    Returns:
        updated_memory_pool: 更新后的内存池
    """
    # 简单的内存池更新逻辑
    if memory_pool is None:
        memory_pool = []
    
    # 添加新数据到内存池
    memory_pool.append(data)
    
    # 保持内存池大小限制（例如最多1000条记录）
    max_size = 1000
    if len(memory_pool) > max_size:
        memory_pool = memory_pool[-max_size:]
    
    return memory_pool