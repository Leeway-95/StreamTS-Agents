import os
import json
import logging
import pandas as pd
from datetime import datetime
from agent_reasoner.memory_pool import MemoryItem
from utils.config import *

logger = logging.getLogger(__name__)


def load_three_parts_from_file(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        parts = [p.strip() for p in content.split('\n\n\n') if p.strip()]
        if len(parts) < 2:
            raise ValueError("Invalid file structure")
        recent_series = parts[0]
        rep_series = parts[1] if len(parts) > 1 else ""
        return recent_series, rep_series,
    except Exception as e:
        logger.error(f"File reading error: {str(e)}")
        return "", ""


def save_full_response(id_val, response, dataset_name, method, task, hist_len, pred_len):
    if dataset_name in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset_name]
    elif dataset_name in DATASET_MERGE_PATHS:
        dataset_path = DATASET_MERGE_PATHS[dataset_name]
    else:
        raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")
    dirname = os.path.dirname(dataset_path)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]
    response_dir = os.path.join(dirname, f"predict-{filename}")
    os.makedirs(response_dir, exist_ok=True)
    response_file = os.path.join(response_dir, f"predict_response.log")

    # 处理不同类型的响应
    if isinstance(response, dict):
        # 如果是字典类型，转换为JSON字符串
        response_str = json.dumps(response, ensure_ascii=False, indent=2)
    else:
        # 如果是字符串类型，直接使用
        response_str = str(response)

    # 构建完整的文件内容，包含Method信息和时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_content = f"\n{'=' * 80}\n"
    file_content += f"Timestamp: {timestamp}\n"
    if method:
        # 根据任务类型决定是否包含HistLen和PredLen
        if task in ["FORECASTING_NUM"]:
            file_content += f"Task: {task} | Dataset: {dataset_name} | Method: {method} | HistLen: {hist_len} | PredLen: {pred_len} | Index: {id_val + 1} | {timestamp}\n"
        else:
            file_content += f"Task: {task} | Dataset: {dataset_name} | Method: {method} | Index: {id_val + 1} | {timestamp}\n"

    else:
        file_content += f"Index: {id_val}\n"
    file_content += f"{'=' * 80}\n"
    file_content += "LLM Response:\n"
    file_content += response_str
    file_content += f"\n"

    # 增量写入模式，追加到文件末尾
    with open(response_file, 'a', encoding="utf-8") as f:
        f.write(file_content)
    # logger.info(f"Full response saved to {response_file}")  # 减少控制台输出，避免中断进度条
    return response_file


def save_log_entry(log_entry, task_type, method):
    try:
        log_entry['LogTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 定义新的列顺序：Index在第一列，不包含ID列
        desired_columns = ['Index', 'Task', 'Dataset', 'Method', 'TTFT', 'InputTokens', 'OutputTokens', 'Cost',
                           'TotalTime', 'ResponseFile', 'LogTime']

        if not os.path.exists(LOG_LLM_METRICS_PATH):
            log_entry['Index'] = 1
            log_df = pd.DataFrame([log_entry])
            # 确保列顺序正确，只包含需要的列
            log_df = log_df.reindex(columns=[col for col in desired_columns if col in log_df.columns])
            log_df.to_csv(LOG_LLM_METRICS_PATH, index=False)
        else:
            existing_df = pd.read_csv(LOG_LLM_METRICS_PATH)
            # 检查 Index 列是否存在，如果不存在则创建
            if 'Index' not in existing_df.columns:
                existing_df['Index'] = range(1, len(existing_df) + 1)
            max_id = existing_df['Index'].max() if len(existing_df) > 0 else 0
            log_entry['Index'] = max_id + 1

            log_df = pd.DataFrame([log_entry])

            # 合并数据
            updated_df = pd.concat([existing_df, log_df], ignore_index=True)
            # 确保列顺序正确，只包含需要的列（去掉ID列）
            updated_df = updated_df.reindex(columns=[col for col in desired_columns if col in updated_df.columns])
            updated_df.to_csv(LOG_LLM_METRICS_PATH, index=False)
        # logger.info(f"Log entries saved to {os.path.basename(LOG_LLM_METRICS_PATH)}, Index: {log_entry['Index']}, Task: {log_entry.get('Task', task_type)}, Method: {log_entry.get('Method', method)}")  # 减少控制台输出，避免中断进度条
    except Exception as e:
        logger.error(f"Log saving error: {str(e)}")


def save_memory_state(memory_pool):
    state = [{'series': item.series, 'position': item.position, 'r_score': item.r_score, 'i_score': item.i_score} for
             item in memory_pool.items]
    with open(Memory_Pool_PATH, 'w') as f:
        json.dump(state, f)


def load_memory_state(memory_pool):
    try:
        if os.path.exists(Memory_Pool_PATH):
            with open(Memory_Pool_PATH, 'r') as f:
                state = json.load(f)
            new_items = []
            for item in state:
                try:
                    new_items.append(MemoryItem(
                        series=item['series'],
                        position=item['position'],
                        r_score=item['r_score'],
                        i_score=item['i_score']
                    ))
                except KeyError as e:
                    logger.error(f"Memory item missing key: {str(e)}")
            memory_pool.items = new_items
            memory_pool.update_threshold()
    except Exception as e:
        logger.error(f"Memory loading error: {str(e)}")
        memory_pool.items = []


def update_csv(id_val, pred_labels, pred_series, impact_scores, dataset_name, task_type, method, hist_len, pred_len):
    try:
        # 检查数据集是否属于当前任务类型
        if task_type == "UNDERSTANDING" and dataset_name not in DATASET_UNDERSTANDING:
            logger.info(f"Skipping generator-5000 {dataset_name} for UNDERSTANDING task")
            return
        elif task_type == "FORECASTING_NUM" and dataset_name not in DATASET_FORECASTING_NUM:
            logger.info(f"Skipping generator-5000 {dataset_name} for FORECASTING_NUM task")
            return
        elif task_type == "FORECASTING_EVENT" and dataset_name not in DATASET_FORECASTING_EVENT:
            logger.info(f"Skipping generator-5000 {dataset_name} for FORECASTING_EVENT task")
            return
        elif task_type == "REASONING" and dataset_name not in DATASET_REASONING:
            logger.info(f"Skipping generator-5000 {dataset_name} for REASONING task")
            return

        # 检查方法是否属于当前任务类型 - 只对Baseline方法进行过滤，OUR_Method不需要过滤
        if method not in OUR_Method:  # 只对非OUR_Method的方法进行过滤
            if task_type == "UNDERSTANDING" and method not in BASELINE_UNDERSTANDING:
                logger.info(f"Skipping method {method} for UNDERSTANDING task")
                return
            elif task_type == "FORECASTING_NUM" and method not in BASELINE_FORECASTING_NUM:
                logger.info(f"Skipping method {method} for FORECASTING_NUM task")
                return

            elif task_type == "FORECASTING_EVENT" and method not in BASELINE_FORECASTING_EVENT:
                logger.info(f"Skipping method {method} for FORECASTING_EVENT task")
                return
            elif task_type == "REASONING" and method not in BASELINE_REASONING:
                logger.info(f"Skipping method {method} for REASONING task")
                return

        # 从DATASET_PATHS中读取路径信息
        # 如果DATASET_PATHS中没有该数据集，尝试从DATASET_MERGE_PATHS中获取
        if dataset_name in DATASET_PATHS:
            dataset_path = DATASET_PATHS[dataset_name]
        elif dataset_name in DATASET_MERGE_PATHS:
            dataset_path = DATASET_MERGE_PATHS[dataset_name]
        else:
            raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")
        dirname = os.path.dirname(dataset_path)
        filename = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = os.path.join(dirname, f"predict-{filename}")
        output_file = os.path.join(output_dir, "predict_summary.csv")

        os.makedirs(output_dir, exist_ok=True)

        # 创建新行数据 - 对于FORECASTING_EVENT任务，不写入Pred_Series
        new_row = {
            "Index": id_val,
            "Task": task_type,
            "Dataset": dataset_name,
            "Method": method,
            "HistLen": int(hist_len),
            "PredLen": int(pred_len),
            "Impact_Scores": json.dumps(impact_scores) if impact_scores else "",
            "Pred_Labels": json.dumps(pred_labels) if pred_labels else ""
        }

        # 只有非FORECASTING_EVENT任务才写入Pred_Series
        if task_type != "FORECASTING_EVENT":
            new_row["Pred_Series"] = json.dumps(pred_series) if pred_series else ""
        else:
            new_row["Pred_Series"] = ""

        # 检查文件是否存在
        if not os.path.exists(output_file):
            # 如果文件不存在，创建新文件并写入列名和第一行数据
            df = pd.DataFrame(
                columns=["Index", "Task", "Dataset", "Method", "HistLen", "PredLen", "Pred_Labels", "Impact_Scores",
                         "Pred_Series"])
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(output_file, index=False)
        else:
            # 如果文件存在，读取现有数据
            try:
                df = pd.read_csv(output_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                df = pd.DataFrame(
                    columns=["Index", "Task", "Dataset", "Method", "HistLen", "PredLen", "Pred_Labels", "Impact_Scores",
                             "Pred_Series"])

            # 确保所有需要的列都存在
            required_columns = ["Index", "Task", "Dataset", "Method", "Pred_Labels", "Impact_Scores", "Pred_Series"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # 重新生成连续的索引，确保不重复
            df['Index'] = range(1, len(df) + 1)
            # 增量写入更新后的数据
            df.to_csv(output_file, index=False)
    except Exception as e:
        logger.error(f"CSV Update Error: {str(e)}")
        raise