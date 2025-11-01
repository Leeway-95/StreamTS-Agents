from utils.common import *
import os
import pandas as pd
import ast
import logging
import json

logger = logging.getLogger(__name__)


def load_questions(dataset_name):
    try:
        dataset_path = DATASET_MERGE_PATHS[dataset_name]
        dirname = os.path.dirname(dataset_path)
        filename = os.path.splitext(os.path.basename(dataset_path))[0]
        input_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")

        if not os.path.exists(input_path):
            logger.warning(f"Questions file not found at {input_path}")
            return [], []

        df = pd.read_csv(input_path)

        # 筛选出对应数据集的数据
        dataset_rows = df[df['Dataset'] == dataset_name]

        if dataset_rows.empty:
            logger.info(f"Available datasets in {dataset_name}: {df['Dataset'].unique().tolist()}")
            return [], []

        # 获取第一行的Question和Positions数据
        first_row = dataset_rows.iloc[0]
        questions_str = first_row['Question']
        positions_str = first_row['Positions']

        # 解析Question列（应该是一个数组字符串）
        try:
            if isinstance(questions_str, str):
                # 尝试使用ast.literal_eval解析
                questions_list = ast.literal_eval(questions_str)
            else:
                questions_list = [questions_str] if questions_str else []
        except (ValueError, SyntaxError):
            # 如果解析失败，将整个字符串作为单个问题
            questions_list = [questions_str] if questions_str else []

        # 解析Positions列
        try:
            if isinstance(positions_str, str):
                positions_list = ast.literal_eval(positions_str)
            else:
                positions_list = []
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse positions for generator-5000 {dataset_name}")
            positions_list = []

        return questions_list, positions_list

    except Exception as e:
        logger.error(f"Error loading QATS-4 questions and positions: {e}")
        return [], []


def format_qats4_questions(questions, positions, full_series):
    if not questions:
        return ""

    formatted_questions = []
    for i, question in enumerate(questions, 1):
        # 清理问题文本，移除多余的引号和转义字符
        clean_question = question.strip().strip('"\'')

        # 获取对应的时间序列范围
        if i <= len(positions):
            position = positions[i - 1]  # positions是0索引的
            if isinstance(position, (list, tuple)) and len(position) == 2:
                clean_question = clean_question.replace("<<Time_Series>>", str(full_series[position[0]:position[1]]))
            else:
                clean_question = clean_question.replace("<<Time_Series>>", str(full_series[position[0]:]))

        formatted_questions.append(f"{i}. {clean_question}")

    return "\n\n".join(formatted_questions)


def get_forecasting_event_prompts(dataset_name, window_size=24, positions=None, id_val=None):
    domain_label = "True/False"
    dataset_instruction = ""
    if dataset_name.startswith("Weather_"):
        # Extract city name from generator-5000 name
        domain_label = '"not rained", "rained"'
        city_name = dataset_name.split("_")[1] if "_" in dataset_name else "the location"
        dataset_instruction = (
            f"Your job is to act as a professional weather forecaster. You will be given a time-series data of the weather from the past 24 hours. Based on this information, your task is to predict whether it will rain in the next 24 hours. "
            f"Your task is to predict whether it will rain or not in {city_name} in the next {window_size} hours. Review the time-series data provided for the last {window_size} hours. Each time-series consists of hourly values separated by a '|' token for the following indicators:"
            f"- Temperature (Kelvin): {{var_1}}"
            f"- Humidity (%): {{var_2}}"
            f"- Air Pressure (hPa): {{var_3}}"
            f"- Wind Speed (m/s): {{var_4}}"
            f"- Wind Direction (degrees): {{var_5}}"
            f"Based on this information, respond with either 'rained' or 'not rained'. Do not provide any other details.\n"
            f"Dataset-specific Instructions:\n- Weather datasets: Predict weather events (e.g., rain/no rain) based on meteorological time series patterns. Analyze temperature, humidity, pressure, and other weather indicators to determine the likelihood of precipitation or other weather phenomena.")

    elif dataset_name.startswith("Finance_"):
        domain_label = '"decrease", "increase", "neutral"'
        # Extract indicator name from generator-5000 name
        indicator_mapping = {
            "Finance_sp500": "S&P 500",
            "Finance_nikkei": "Nikkei 225"
        }
        indicator_name = indicator_mapping.get(dataset_name, "financial indicator")

        dataset_instruction = f"Your job is to act as a professional financial forecaster. You will be given a time-series data from the past 20 market days. Based on this information, your task is to predict whether the {indicator_name} price will decrease by more than 1%, increase by more than 1%, or change minimally in the next market day."

        dataset_instruction += f"""Your task is to predict whether the {indicator_name} price will: (1) Decrease: decrease by more than 1% (2) Increase: increase by more than 1% (3) Neutral: change minimally, between -1% to 1%
        in the next market day. Review the time-series data provided for the last {window_size} market days. Each time-series consists of daily values separated by a '|' token for the following indicators:
        - S&P 500: {{var_1}}
        - VIX (Volatility Index): {{var_2}}
        - Nikkei 225: {{var_3}}
        - FTSE 100: {{var_4}}
        - Gold Futures: {{var_5}}
        - Crude Oil Futures: {{var_6}}
        - Exchange rate for EUR/USD: {{var_7}}
        - Exchange rate for USD/JYP: {{var_8}}
        - Exchange rate for USD/CNY: {{var_9}}
        Based on this information, predict whether the {indicator_name} price will decrease by more than 1%, increase by more than 1%, or otherwise, in the next market day. Respond with either 'decrease', 'increase', or 'neutral'. Do not provide any other details.\n
        Dataset-specific Instructions:\n- Finance datasets: Predict financial market events (e.g., significant price movements, market volatility) based on market indicators. Analyze price trends, volume patterns, and market dynamics to identify potential market events or anomalies."""

    elif dataset_name.startswith("Healthcare_"):
        domain_label = '"did not exceed the average", "exceeded the average"'
        if "mortality" in dataset_name.lower():
            dataset_instruction = "Your job is to act as a professional healthcare forecaster. You will be given a time-series data from the past 20 weeks. Based on this information, your task is to predict whether the ratio of mortality from Influenza or Pneumonia to the total number of deaths will exceed its average in the coming week."

            dataset_instruction += f"""Your task is to predict whether the ratio of mortality from Influenza or Pneumonia to the total number of deaths will: (1) Exceed its average (2) Not exceed its average in the coming week. Review the time-series data provided for the last {window_size} weeks. Each time-series consists of weekly values separated by a '|' token for the following indicators:
            - Total deaths: {{var_1}}
            - Influenza/Pneumonia deaths: {{var_2}}/{{var_3}}
            - Mortality ratio (%): {{var_4}}
            Based on this time-series data, predict whether the mortality ratio will exceed its average or not in the coming week. Respond with either 'exceeded the average' or 'did not exceed the average'. Do not provide any other details."""

        else:  # Healthcare_positive
            dataset_instruction = "Your job is to act as a professional healthcare forecaster. You will be given a time-series data from the past 20 weeks. Based on this information, your task is to predict whether the percentage of respiratory specimens testing positive for influenza will exceed its average of 6.26% in the coming week."

            dataset_instruction += f"""Your task is to predict whether the percentage of respiratory specimens testing positive for influenza will: (1) Exceed its average of 6.26% (2) Not exceed its average of 6.26% in the coming week. Review the time-series data provided for the last {window_size} weeks. Each time-series consists of weekly values separated by a '|' token for the following indicators:
            - Number of specimens tested: {{var_1}}
            - Number of positive specimens for Influenza A: {{var_2}}
            - Number of positive specimens for Influenza B: {{var_3}}
            - Ratio of positive specimens (%): {{var_4}}
            - Ratio of positive specimens for Influenza A (%): {{var_5}}
            - Ratio of positive specimens for Influenza B (%): {{var_6}}
            Based on this time-series data, predict whether the percentage of respiratory specimens testing positive for influenza will exceed its average of 6.26% or not in the coming week. Respond with either 'exceeded the average' or 'not did not exceed the average'. Do not provide any other details."""
        dataset_instruction += "Dataset-specific Instructions:\n- Healthcare datasets: Predict healthcare outcomes (e.g., patient mortality, positive test results) based on patient monitoring data patterns. Analyze vital signs, laboratory values, and clinical indicators to assess patient risk and predict health events."

    # 去掉所有缩进
    dataset_instruction = dataset_instruction.replace("\n            ", "\n").replace("\n        ", "\n")

    # 加载对应数据集数据的历史Series填充到dataset_instruction的{{var_*}}中
    # 从DATASET_PATHS或DATASET_MERGE_PATHS中读取路径信息
    if dataset_name in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset_name] + ".csv"
    elif dataset_name in DATASET_MERGE_PATHS:
        dataset_path = DATASET_MERGE_PATHS[dataset_name] + ".csv"
    else:
        raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")

    dirname = os.path.dirname(dataset_path)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]
    stream_summary_path = os.path.join(dirname, f"stream-{filename}", "stream_summary.csv")

    if os.path.exists(stream_summary_path):
        df = pd.read_csv(stream_summary_path)
        dataset_rows = df[df['Dataset'] == dataset_name]

        if not dataset_rows.empty:
            processed_instruction = dataset_instruction

            # 收集所有行的变量数据，建立变量到数据的映射
            variable_data_map = {}

            # 遍历数据集的所有行来收集变量数据
            for row_idx, row in dataset_rows.iterrows():
                # 获取当前行的Positions数据
                positions_data = []
                if 'Positions' in row and row['Positions']:
                    try:
                        positions_data = ast.literal_eval(row['Positions'])
                    except (ValueError, SyntaxError):
                        logger.warning(f"Failed to parse positions for generator-5000 {dataset_name}, row {row_idx}")

                # 获取当前行的Variable数据
                row_variables = []
                if 'Variable' in row and row['Variable']:
                    try:
                        row_variables = ast.literal_eval(row['Variable']) if isinstance(row['Variable'], str) else row[
                            'Variable']
                        if not isinstance(row_variables, list):
                            row_variables = [row_variables]
                    except (ValueError, SyntaxError):
                        row_variables = [row['Variable']]

                # 获取当前行的Series数据
                # 安全解析Series数据，处理NaN值
                if 'Series' in row and row['Series'] and str(row['Series']).strip():
                    try:
                        # 首先尝试使用json.loads解析
                        row_series_data = json.loads(row['Series'])
                    except json.JSONDecodeError:
                        try:
                            # 如果json.loads失败，尝试使用ast.literal_eval解析
                            row_series_data = ast.literal_eval(row['Series'])
                        except (ValueError, SyntaxError):
                            logger.warning(
                                f"Unable to parse Series field in prompt_builder: {str(row['Series'])[:50]}...")
                            # 如果包含NaN值，尝试手动处理
                            try:
                                import numpy as np
                                # 将NaN替换为None或0，然后解析
                                series_str = str(row['Series']).replace('nan', 'null')
                                row_series_data = json.loads(series_str)
                                # 将null转换为0或适当的数值
                                row_series_data = [0 if x is None else x for x in row_series_data]
                            except:
                                logger.warning(f"Failed to parse Series data in prompt_builder, using empty list")
                                row_series_data = []
                else:
                    row_series_data = []

                # 为每个变量建立数据映射
                for var_idx, var_name in enumerate(row_variables, 1):
                    # 修复：使用Positions中最大的范围截取有效数据
                    if positions_data and len(positions_data) > 0:
                        # 找到所有位置区间中的最大范围
                        max_start = float('inf')
                        max_end = 0

                        for pos_range in positions_data:
                            if isinstance(pos_range, (list, tuple)) and len(pos_range) >= 2:
                                start, end = pos_range[0], pos_range[1]
                                max_start = min(max_start, start)
                                max_end = max(max_end, end)

                        # 如果找到了有效的范围，使用最大范围截取数据
                        if max_start != float('inf') and max_end > max_start:
                            # 确保索引在有效范围内
                            max_start = max(0, max_start)
                            max_end = min(len(row_series_data), max_end)
                            series_str = '|'.join(map(str, row_series_data[max_start:max_end]))
                        else:
                            # 如果位置范围不完整，使用全部数据
                            series_str = '|'.join(map(str, row_series_data))
                    else:
                        # 如果没有positions_data，使用整个序列
                        series_str = '|'.join(map(str, row_series_data))

                    # 将变量数据存储到映射中，使用变量名作为key
                    if var_name not in variable_data_map:
                        variable_data_map[var_name] = series_str

            # 替换所有{{var_*}}占位符
            # 按照var_1, var_2, var_3...的顺序进行替换
            for var_idx in range(1, 10):  # 支持最多9个变量
                var_placeholder = f"{{var_{var_idx}}}"
                if var_placeholder in processed_instruction:
                    # 查找对应的变量名（通常是var_1, var_2等）
                    var_name = f"var_{var_idx}"
                    if var_name in variable_data_map:
                        processed_instruction = processed_instruction.replace(var_placeholder,
                                                                              variable_data_map[var_name])
                    else:
                        # 如果没有找到对应的变量，尝试使用第一个可用的变量数据
                        if variable_data_map:
                            first_available_data = list(variable_data_map.values())[0]
                            processed_instruction = processed_instruction.replace(var_placeholder, first_available_data)

            dataset_instruction = processed_instruction

    # 构建完整的数据集特定指令
    dataset_instruction += """TASK EXECUTION:
- Act according to the system prompt role and expertise
- Follow the user prompt template structure for analysis
- Provide predictions in the exact format specified\n"""

    return dataset_instruction, domain_label


def build_caption_guided_cot_prompt(reasoning_task, temporal_range, caption, cot_content, context_playbook):
    """
    构建Caption-guided-CoT提示
    
    Args:
        reasoning_task: 推理任务描述
        temporal_range: 时间范围
        caption: Caption数据列表
        cot_content: CoT指令内容
        context_playbook: Playbook上下文
        
    Returns:
        str: 构建好的提示
    """
    template_path = os.path.join(os.path.dirname(__file__), "CoT", "Caption-Guided-CoT.txt")
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        # 替换模板变量
        prompt = template.replace("{{ reasoning_task }}", reasoning_task)
        prompt = prompt.replace("{{ temporal_range }}", temporal_range)
        prompt = prompt.replace("{{ caption }}", json.dumps(caption, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{{ CoT }}", cot_content)
        prompt = prompt.replace("{{ context_playbook }}", context_playbook)
        
        return prompt
    except FileNotFoundError:
        logger.error(f"Template file not found at {template_path}")
        return ""


def load_caption_data(dataset_name, temporal_range):
    """
    从caption_storage系统加载caption数据
    
    Args:
        dataset_name: 数据集名称
        temporal_range: 时间范围字符串
        
    Returns:
        list: caption数据列表
    """
    # 首先生成默认的智能captions作为基础
    def generate_smart_captions(dataset_name, temporal_range):
        """生成智能的默认captions"""
        if isinstance(temporal_range, str):
            import re
            numbers = re.findall(r'\d+', temporal_range)
            if len(numbers) >= 2:
                start_time, end_time = int(numbers[0]), int(numbers[1])
                range_size = end_time - start_time
                
                # 根据范围大小和数据集生成不同的captions
                captions = [
                    f"Time series data analysis for {dataset_name} in temporal range {temporal_range}",
                    f"Pattern analysis covering {range_size} time points from {start_time} to {end_time}",
                ]
                
                # 根据范围大小添加特定的分析描述
                if range_size > 10000:
                    captions.append("Long-term trend analysis with potential seasonal patterns")
                elif range_size > 1000:
                    captions.append("Medium-term pattern detection including trend and volatility analysis")
                else:
                    captions.append("Short-term fluctuation analysis focusing on immediate patterns")
                
                return captions
        
        return [
            f"Time series analysis in range {temporal_range}",
            f"Temporal patterns detected in {dataset_name} dataset",
            f"Statistical analysis of sequential data patterns"
        ]
    
    # 尝试从caption_storage系统获取数据
    try:
        import sys
        import os
        caption_storage_path = os.path.join(os.path.dirname(__file__), '..', 'caption_storage')
        if caption_storage_path not in sys.path:
            sys.path.append(caption_storage_path)
        
        from caption_block import CaptionBlockSystem
        
        # 使用caption_storage目录作为数据目录
        caption_data_dir = os.path.join(caption_storage_path, "output")
        caption_index_dir = os.path.join(caption_storage_path, "index_cache")
        
        # 初始化caption系统
        caption_system = CaptionBlockSystem(data_dir=caption_data_dir, index_dir=caption_index_dir)
        caption_system.build_index()
        
        # 解析时间范围
        if isinstance(temporal_range, str):
            # 从"[start, end]"格式中提取数字
            import re
            numbers = re.findall(r'\d+', temporal_range)
            if len(numbers) >= 2:
                start_time, end_time = int(numbers[0]), int(numbers[1])
            else:
                start_time, end_time = 0, 1000
        else:
            start_time, end_time = 0, 1000
        
        # 构建多种查询策略
        queries = [
            f"时间范围[{start_time}, {end_time}]",
            f"range {start_time} {end_time}",
            f"upward trend {temporal_range}",
            f"trend analysis {temporal_range}",
            f"temporal pattern {start_time} {end_time}"
        ]
        
        all_captions = []
        for query in queries:
            try:
                logger.debug(f"Searching with query: {query}")
                results = caption_system.search(query)
                logger.debug(f"Search returned {len(results)} results")
                
                if results:
                    for result in results:
                        if isinstance(result, dict):
                            # 提取caption字段
                            caption_text = result.get('caption', '') or result.get('description', '')
                            if caption_text and caption_text not in all_captions:
                                all_captions.append(caption_text)
                                logger.debug(f"Added caption: {caption_text[:100]}...")
                        elif isinstance(result, str):
                            if result not in all_captions:
                                all_captions.append(result)
                                logger.debug(f"Added string result: {result[:100]}...")
                                
                # 如果已经找到足够的captions，停止搜索
                if len(all_captions) >= 3:
                    break
                    
            except Exception as query_error:
                logger.debug(f"Query '{query}' failed: {query_error}")
                continue
        
        # 如果获取到了captions，返回前5个
        if all_captions:
            logger.info(f"Successfully loaded {len(all_captions)} captions from caption_storage")
            return all_captions[:5]
        else:
            # 如果没有获取到，使用智能默认captions
            logger.info("No captions found in caption_storage, using smart default captions")
            return generate_smart_captions(dataset_name, temporal_range)
            
    except ImportError as e:
        logger.info(f"Caption storage system not available: {e}, using smart default captions")
        return generate_smart_captions(dataset_name, temporal_range)
    except Exception as e:
        logger.warning(f"Error loading caption data: {e}, using smart default captions")
        return generate_smart_captions(dataset_name, temporal_range)


def extract_cot_content_from_original(cot_file_path):
    """
    从原始CoT文件中提取内容，去掉不需要的部分
    
    Args:
        cot_file_path: 原始CoT文件路径
        
    Returns:
        str: 提取的CoT内容
    """
    try:
        with open(cot_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 去掉不需要的内容
        lines = content.split('\n')
        filtered_lines = []
        skip_section = False
        
        for line in lines:
            # 跳过以下内容：
            # 1. Your role is a time series analysis expert
            # 2. INPUT: 相关部分
            # 3. SubSequence 相关部分
            # 4. 图片相关内容
            if any(skip_phrase in line for skip_phrase in [
                'Your role is a time series analysis expert',
                'INPUT:',
                'SubSequence',
                'Subsequence',
                'Visual Integration',
                'If plots are provided',
                'integrate visual pattern recognition',
                'X-axis = time; Y-axis = values',
                'Red line = actual sequence',
                'Use plots to identify patterns'
            ]):
                skip_section = True
                continue
            elif line.startswith('Goal') or line.startswith('FINAL OUTPUT') or line.startswith('Enhanced') or line.startswith('Critical') or line.startswith('Temporal Patterns'):
                skip_section = False
            
            if not skip_section and line.strip():
                # 清理占位符和格式，但保留有用内容
                if not any(placeholder in line for placeholder in ['<<', '>>']):
                    # 进一步过滤图片相关内容
                    if not any(visual_term in line.lower() for visual_term in [
                        'visual', 'plot', 'chart', 'graph', 'image', 'figure'
                    ]):
                        filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    except FileNotFoundError:
        return ""


def build_pcot_prompt(id_val, full_series, recent_series, rep_series, memory_pool, dataset_name, method, pred_len,
                      hist_len, positions):

    # 根据数据集名称确定任务类型
    task_type = None
    for task in TASK:
        if task == "UNDERSTANDING" and dataset_name in DATASET_UNDERSTANDING:
            task_type = "UNDERSTANDING"
            break
        elif task == "REASONING" and dataset_name in DATASET_REASONING:
            task_type = "REASONING"
            break
        elif task == "FORECASTING_NUM" and dataset_name in DATASET_FORECASTING_NUM:
            task_type = "FORECASTING_NUM"
            break
        elif task == "FORECASTING_EVENT" and dataset_name in DATASET_FORECASTING_EVENT:
            task_type = "FORECASTING_EVENT"
            break

    # 根据当前方法和任务类型选择对应的提示文件
    cot_file = ""
    pcot_input = ""
    memory_patches = []
    if method == "StreamTS-Agents" or method == "StreamTS-Agents (+v)":
        # 限制代表性序列的数量
        rep_lines = rep_series.split('\n')[:MEM_TOP_K] if rep_series else []
        formatted_reps = []

        # 格式化代表性序列
        for i, line in enumerate(rep_lines):
            if line.strip():
                formatted_reps.append(f"R_{i + 1}: {line}")

        # 不再添加"Your role is a time series analysis expert"和"INPUT:"部分
        # 这些内容会在Caption-guided-CoT模板中处理
        pcot_input = ""

        # 获取内存池中的记忆项
        memory_patches = memory_pool.get_memory_patches()
        # 根据任务类型选择对应的提示文件
        if task_type == "UNDERSTANDING":
            cot_file = Prompt_PATHS["StreamTS-Agents-Understand"]
        elif task_type == "REASONING":
            cot_file = Prompt_PATHS["StreamTS-Agents-Reason"]
        elif task_type == "FORECASTING_EVENT":
            cot_file = Prompt_PATHS["StreamTS-Agents-Forecast-Event"]
        else:
            cot_file = Prompt_PATHS["StreamTS-Agents-Reason"]
    elif method == "PromptCast":
        cot_file = Prompt_PATHS["PromptCast"]
    elif method == "TimeCP":
        cot_file = Prompt_PATHS["TimeCP"]
    elif method == "TimeCAP":
        cot_file = Prompt_PATHS["TimeCAP"]
    elif method == "Inf-LLM" or method == "Inf-LLM (+v)":
        if task_type == "UNDERSTANDING":
            cot_file = Prompt_PATHS["Inf-LLM-Understand"]
        elif task_type == "REASONING":
            cot_file = Prompt_PATHS["Inf-LLM-Reason"]
    elif method == "Window" or method == "Window (+v)":
        if task_type == "UNDERSTANDING":
            cot_file = Prompt_PATHS["Window-Understand"]
        elif task_type == "REASONING":
            cot_file = Prompt_PATHS["Window-Reason"]
    else:
        cot_file = Prompt_PATHS["Inf-LLM-Reason"]

    # 从原始CoT文件提取内容（去掉INPUT: SubSequence部分）
    cot_content = extract_cot_content_from_original(cot_file)
    
    # 处理CoT内容中的占位符
    if "<<Domain>>" in cot_content:
        if dataset_name == "Gold":
            domain = "financial"
        elif dataset_name.startswith("ETTm"):
            domain = "electricity"
        elif dataset_name == "Weather":
            domain = "weather"
        else:
            domain = "data"
        cot_content = cot_content.replace("<<Domain>>", domain)
    
    if "<<Questions>>" in cot_content:
        # 加载QATS-4问题和位置信息
        qats4_questions, qats4_positions = load_questions(dataset_name)
        formatted_questions = format_qats4_questions(qats4_questions, qats4_positions, full_series)
        cot_content = cot_content.replace("<<Questions>>", formatted_questions)
    
    if "<<PreLen>>" in cot_content:
        cot_content = cot_content.replace("<<PreLen>>", str(pred_len))
    if "<<Histlen>>" in cot_content:
        cot_content = cot_content.replace("<<Histlen>>", str(hist_len))
    if "<<Time_Series>>" in cot_content:
        cot_content = cot_content.replace("<<Time_Series>>", str(recent_series))
    if "<<Fulllen>>" in cot_content:
        cot_content = cot_content.replace("<<Fulllen>>", str(len(full_series)))
    if "<<Positions>>" in cot_content:
        cot_content = cot_content.replace("<<Positions>>", str(positions))
    if "<<Poslen>>" in cot_content:
        cot_content = cot_content.replace("<<Poslen>>", str(len(positions)))
    if "<<MLen>>" in cot_content:
        cot_content = cot_content.replace("<<MLen>>", str(len(memory_patches)))
    if "<<RSLen>>" in cot_content:
        cot_content = cot_content.replace("<<RSLen>>", str(len(rep_series)))

    # 处理事件预测的数据集特定指令
    if "<<DATASET_SPECIFIC_INSTRUCTION>>" in cot_content:
        # 使用新的函数获取数据集特定的系统和用户提示
        if task_type == "FORECASTING_EVENT":
            dataset_instruction, domain_label = get_forecasting_event_prompts(dataset_name, hist_len, positions, id_val)
            cot_content = cot_content.replace("<<DATASET_SPECIFIC_INSTRUCTION>>", dataset_instruction)
            # 确保 Domain_Label 被正确替换为实际的标签值
            if "<<Domain_Label>>" in cot_content:
                cot_content = cot_content.replace("<<Domain_Label>>", str(domain_label))

    # 构建时间范围字符串
    if positions and len(positions) > 0:
        # 从positions推断时间范围
        min_pos = min(pos[0] for pos in positions)
        max_pos = max(pos[1] for pos in positions)
        temporal_range = f"[{min_pos}, {max_pos}]"
    else:
        temporal_range = "[unknown, unknown]"
    
    # 加载caption数据
    caption_data = load_caption_data(dataset_name, temporal_range)
    
    # 构建推理任务描述
    if task_type == "UNDERSTANDING":
        reasoning_task = f"Analyze and understand the temporal patterns in {dataset_name} dataset within range {temporal_range}"
    elif task_type == "REASONING":
        reasoning_task = f"Perform reasoning analysis on {dataset_name} dataset within range {temporal_range}"
    elif task_type == "FORECASTING_NUM":
        reasoning_task = f"Forecast numerical values for {dataset_name} dataset within range {temporal_range}"
    elif task_type == "FORECASTING_EVENT":
        reasoning_task = f"Forecast events for {dataset_name} dataset within range {temporal_range}"
    else:
        reasoning_task = f"Analyze {dataset_name} dataset within range {temporal_range}"
    
    # 获取playbook上下文
    from playbook_manager import get_playbook_context_string
    context_playbook = get_playbook_context_string()
    
    # 使用Caption-guided-CoT模板构建最终提示
    content = build_caption_guided_cot_prompt(
        reasoning_task=reasoning_task,
        temporal_range=temporal_range,
        caption=caption_data,
        cot_content=cot_content,
        context_playbook=context_playbook
    )

    if method == "PromptCast":
        # PromptCast特定替换
        content = content.replace("<<Start_Time>>", str(id_val))
        content = content.replace("<<End_Time>>", str(id_val + hist_len))
        if dataset_name == "Gold":
            dataset_variable = "gold price (USD per ounce)"
        elif dataset_name.startswith("ETTm"):
            dataset_variable = "electricity (MW)"
        elif dataset_name == "Weather":
            dataset_variable = "weather (percent)"
        else:
            dataset_variable = "value"
        content = content.replace("<<dataset_variable>>", dataset_variable)
    elif method == "TimeCP":
        if dataset_name == "Gold":
            domain_variable = "gold price"
            prediction_type = "price change"
        elif dataset_name.startswith("ETTm"):
            domain_variable = "electricity"
            prediction_type = "change in electricity"
        elif dataset_name == "Weather":
            domain_variable = "weather"
            prediction_type = "whether it will rain"
        else:
            domain_variable = "value"
            prediction_type = "value"
        content = content.replace("<<domain_variable>>", domain_variable)
        content = content.replace("<<prediction_type>>", prediction_type)
    elif method == "Inf-LLM" or method == "Inf-LLM (+v)":
        memory_patches = memory_pool.get_memory_patches()
        # 将列表转换为字符串，每个项目占一行
        memory_patches_str = "\n".join(memory_patches) if memory_patches else ""
        content = content.replace("<<Memory_Patches>>", memory_patches_str)
    
    pcot_input += content

    # 从DATASET_PATHS或DATASET_MERGE_PATHS中读取路径信息
    if dataset_name in DATASET_PATHS:
        dataset_path = DATASET_PATHS[dataset_name] + ".csv"
    elif dataset_name in DATASET_MERGE_PATHS:
        dataset_path = DATASET_MERGE_PATHS[dataset_name] + ".csv"
    else:
        raise KeyError(f"Dataset path not found for {dataset_name} in either DATASET_PATHS or DATASET_MERGE_PATHS")
    dirname = os.path.dirname(dataset_path)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]

    # 构建图像目录路径
    image_dir = os.path.join(dirname, f"detection-{filename}/series_{id_val}")
    images = []

    # 只有当方法包含(+v)时才收集图像
    if has_vision_support(method):
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                if filename.endswith(".png"):
                    file_path = os.path.join(image_dir, filename)
                    images.append(file_path)

    return pcot_input, images


def build_caption_guided_cot_prompt(reasoning_task, temporal_range, caption, cot_content, context_playbook):
    """
    构建Caption-guided-CoT模板的prompt
    
    Args:
        reasoning_task: 推理任务描述
        temporal_range: 时间范围
        caption: 标题/描述信息
        cot_content: CoT模板内容
        context_playbook: 上下文playbook信息
    
    Returns:
        str: 格式化后的prompt内容
    """
    # 读取Caption-guided-CoT.txt模板
    # 尝试不同的路径
    template_paths = [
        "agent_reasoner/CoT/Caption-Guided-CoT.txt",  # 从项目根目录运行
        "CoT/Caption-Guided-CoT.txt",  # 从agent_reasoner目录运行
        os.path.join(os.path.dirname(__file__), "CoT", "Caption-Guided-CoT.txt")  # 相对于当前文件
    ]
    
    template_path = None
    for path in template_paths:
        if os.path.exists(path):
            template_path = path
            break
    
    if not template_path:
        logger.error(f"Caption-guided-CoT template not found in any of: {template_paths}")
        return ""
    
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
            
        # 替换模板变量
        template_content = template_content.replace("{{ reasoning_task }}", str(reasoning_task))
        template_content = template_content.replace("{{ temporal_range }}", str(temporal_range))
        template_content = template_content.replace("{{ caption }}", str(caption))
        template_content = template_content.replace("{{ CoT }}", str(cot_content))
        template_content = template_content.replace("{{ context_playbook }}", str(context_playbook))
        
        return template_content
        
    except FileNotFoundError:
        logger.error(f"Caption-guided-CoT template not found at {template_path}")
        return ""
    except Exception as e:
        logger.error(f"Error processing Caption-guided-CoT template: {e}")
        return ""


def load_caption_data(dataset_name, temporal_range):
    """
    从caption_storage目录加载指定时间范围的caption数据
    
    Args:
        dataset_name: 数据集名称
        temporal_range: 时间范围，格式如[start, end]
    
    Returns:
        list: caption数据列表
    """
    try:
        # 构建caption存储路径
        if dataset_name in DATASET_PATHS:
            dataset_path = DATASET_PATHS[dataset_name]
        elif dataset_name in DATASET_MERGE_PATHS:
            dataset_path = DATASET_MERGE_PATHS[dataset_name]
        else:
            logger.warning(f"Dataset path not found for {dataset_name}")
            return []
            
        dirname = os.path.dirname(dataset_path)
        caption_dir = os.path.join(dirname, "caption_storage")
        
        if not os.path.exists(caption_dir):
            logger.warning(f"Caption storage directory not found: {caption_dir}")
            return []
            
        # 根据时间范围加载对应的caption文件
        captions = []
        if isinstance(temporal_range, (list, tuple)) and len(temporal_range) >= 2:
            start_time, end_time = temporal_range[0], temporal_range[1]
            
            # 查找时间范围内的caption文件
            for filename in os.listdir(caption_dir):
                if filename.endswith(".json"):
                    caption_file = os.path.join(caption_dir, filename)
                    try:
                        with open(caption_file, "r", encoding="utf-8") as f:
                            caption_data = json.load(f)
                            
                        # 检查caption是否在指定时间范围内
                        if "timestamp" in caption_data:
                            timestamp = caption_data["timestamp"]
                            if start_time <= timestamp <= end_time:
                                captions.append(caption_data.get("caption", ""))
                                
                    except Exception as e:
                        logger.warning(f"Error loading caption file {filename}: {e}")
                        
        return captions
        
    except Exception as e:
        logger.error(f"Error loading caption data: {e}")
        return []


def load_playbook_cache(cache_path="playbook_cache.json"):
    """
    加载playbook缓存数据
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        dict: playbook缓存数据
    """
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
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


def save_playbook_cache(playbook_data, cache_path="playbook_cache.json"):
    """
    保存playbook缓存数据
    
    Args:
        playbook_data: playbook数据
        cache_path: 缓存文件路径
    """
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(playbook_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving playbook cache: {e}")