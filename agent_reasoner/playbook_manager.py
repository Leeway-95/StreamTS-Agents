import json
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def extract_playbook_updates(llm_response: str) -> Optional[Dict[str, Any]]:
    """
    从LLM响应中提取playbook_updates部分
    
    Args:
        llm_response: LLM的JSON响应字符串
        
    Returns:
        dict: playbook_updates数据，如果提取失败返回None
    """
    try:
        response_data = json.loads(llm_response)
        
        if "playbook_updates" in response_data:
            return response_data["playbook_updates"]
        else:
            logger.warning("No playbook_updates found in LLM response")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Error extracting playbook updates: {e}")
        return None


def load_playbook_cache(cache_path="playbook_cache.md"):
    """
    加载playbook缓存数据（Markdown格式）
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        dict: playbook缓存数据
    """
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 解析Markdown格式的playbook
            playbook_data = {
                "insights": [],
                "experiences": [],
                "best_practices": []
            }
            
            current_section = None
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('## Insights'):
                    current_section = 'insights'
                elif line.startswith('## Experiences'):
                    current_section = 'experiences'
                elif line.startswith('## Best Practices'):
                    current_section = 'best_practices'
                elif line.startswith('- ') and current_section:
                    item = line[2:].strip()  # 移除"- "前缀
                    if item:
                        playbook_data[current_section].append(item)
            
            return playbook_data
        else:
            # 如果缓存文件不存在，返回空的playbook结构
            return {
                "insights": [],
                "experiences": [],
                "best_practices": []
            }
    except Exception as e:
        logger.error(f"Error loading playbook cache: {e}")
        return {
            "insights": [],
            "experiences": [],
            "best_practices": []
        }


def save_playbook_cache(playbook_data, cache_path="playbook_cache.md"):
    """
    保存playbook缓存数据（Markdown格式）
    
    Args:
        playbook_data: playbook数据
        cache_path: 缓存文件路径
    """
    try:
        content = "# Playbook Cache\n\n"
        content += "This file contains accumulated insights, experiences, and best practices from LLM reasoning tasks.\n\n"
        
        # 生成Insights部分
        content += "## Insights\n\n"
        if playbook_data.get("insights"):
            for insight in playbook_data["insights"]:
                content += f"- {insight}\n"
        else:
            content += "No insights available yet.\n"
        
        # 生成Experiences部分
        content += "\n## Experiences\n\n"
        if playbook_data.get("experiences"):
            for experience in playbook_data["experiences"]:
                content += f"- {experience}\n"
        else:
            content += "No experiences available yet.\n"
        
        # 生成Best Practices部分
        content += "\n## Best Practices\n\n"
        if playbook_data.get("best_practices"):
            for practice in playbook_data["best_practices"]:
                content += f"- {practice}\n"
        else:
            content += "No best practices available yet.\n"
        
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    except Exception as e:
        logger.error(f"Error saving playbook cache: {e}")


def update_playbook_cache(new_updates: Dict[str, Any], cache_path: str = "playbook_cache.md") -> bool:
    """
    更新playbook缓存，合并新的insights、experiences和best_practices
    
    Args:
        new_updates: 新的playbook更新数据
        cache_path: 缓存文件路径
        
    Returns:
        bool: 更新是否成功
    """
    try:
        # 加载现有缓存
        existing_cache = load_playbook_cache(cache_path)
        
        # 合并新数据
        if "insights" in new_updates and isinstance(new_updates["insights"], list):
            existing_cache["insights"].extend(new_updates["insights"])
            
        if "experiences" in new_updates and isinstance(new_updates["experiences"], list):
            existing_cache["experiences"].extend(new_updates["experiences"])
            
        if "best_practices" in new_updates and isinstance(new_updates["best_practices"], list):
            existing_cache["best_practices"].extend(new_updates["best_practices"])
        
        # 去重处理（保持顺序）
        existing_cache["insights"] = list(dict.fromkeys(existing_cache["insights"]))
        existing_cache["experiences"] = list(dict.fromkeys(existing_cache["experiences"]))
        existing_cache["best_practices"] = list(dict.fromkeys(existing_cache["best_practices"]))
        
        # 保存更新后的缓存
        save_playbook_cache(existing_cache, cache_path)
        
        logger.info(f"Successfully updated playbook cache with {len(new_updates.get('insights', []))} insights, "
                   f"{len(new_updates.get('experiences', []))} experiences, "
                   f"{len(new_updates.get('best_practices', []))} best practices")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating playbook cache: {e}")
        return False


def get_playbook_context_string(cache_path: str = "playbook_cache.md") -> str:
    """
    获取playbook上下文的字符串表示，用于模板替换
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        str: 格式化的playbook上下文字符串
    """
    try:
        cache_data = load_playbook_cache(cache_path)
        
        context_parts = []
        
        if cache_data.get("insights"):
            context_parts.append("Insights:")
            for i, insight in enumerate(cache_data["insights"][-10:], 1):  # 最近10条
                context_parts.append(f"  {i}. {insight}")
                
        if cache_data.get("experiences"):
            context_parts.append("Experiences:")
            for i, experience in enumerate(cache_data["experiences"][-10:], 1):  # 最近10条
                context_parts.append(f"  {i}. {experience}")
                
        if cache_data.get("best_practices"):
            context_parts.append("Best Practices:")
            for i, practice in enumerate(cache_data["best_practices"][-10:], 1):  # 最近10条
                context_parts.append(f"  {i}. {practice}")
        
        return "\n".join(context_parts) if context_parts else "No previous playbook context available."
        
    except Exception as e:
        logger.error(f"Error getting playbook context string: {e}")
        return "Error loading playbook context."


def process_llm_response_with_playbook(llm_response: str, dataset_name: str = None) -> Dict[str, Any]:
    """
    处理LLM响应，提取并更新playbook信息
    
    Args:
        llm_response: LLM响应字符串
        dataset_name: 数据集名称（可选）
        
    Returns:
        dict: 处理后的响应数据
    """
    try:
        # 解析LLM响应
        response_data = json.loads(llm_response)
        
        # 提取playbook更新
        playbook_updates = extract_playbook_updates(llm_response)
        
        if playbook_updates:
            # 更新playbook缓存
            update_success = update_playbook_cache(playbook_updates)
            
            if update_success:
                logger.info("Playbook cache updated successfully")
            else:
                logger.warning("Failed to update playbook cache")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing LLM response with playbook: {e}")
        return {}