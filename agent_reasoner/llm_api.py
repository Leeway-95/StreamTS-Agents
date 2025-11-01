import requests
import time
import logging
import base64
from utils.common import *

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM API客户端类，用于与大语言模型API进行交互
    """

    def __init__(self, api_key=LLM_API_KEY, host=LLM_HOST, api_path=API_PATH):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            host: API主机地址
            api_path: API路径
        """
        self.api_key = api_key
        self.base_host = host
        self.api_path = api_path
        self.full_url = f"{self.base_host}{self.api_path}"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "JD-LLM-Client/1.0",
            "Accept": "application/json"
        })

    def _send_request(self, payload, retries=3, backoff_factor=0.5):
        """
        发送请求到API，支持重试机制

        Args:
            payload: 请求负载
            retries: 重试次数
            backoff_factor: 退避因子

        Returns:
            API响应结果
        """
        if payload is None:
            logger.error("Payload cannot be None")
            return {"error": "Payload cannot be None"}
            
        for attempt in range(retries):
            try:
                response = self.session.post(url=self.full_url, json=payload, timeout=60)
                if response is None:
                    logger.error("Received None response")
                    return {"error": "Received None response from server"}
                    
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        return json_response if json_response is not None else {"error": "Empty JSON response"}
                    except Exception as e:
                        logger.error(f"Failed to parse JSON response: {str(e)}")
                        return {"error": f"Invalid JSON response: {str(e)}"}
                elif response.status_code == 404:
                    alt_path = self._get_alternative_path()
                    self.api_path = alt_path
                    self.full_url = f"{self.base_host}{self.api_path}"
                    continue
                elif response.status_code == 502:
                    if attempt < retries - 1:
                        sleep_time = backoff_factor * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                    else:
                        return {"error": "502 Bad Gateway - Server unavailable", "details": response.text or "No response text"}
                else:
                    return {
                        "error": f"API error: {response.status_code}",
                        "status": response.status_code,
                        "response": response.text or "No response text"
                    }
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(backoff_factor * (2 ** attempt))
                    continue
                return {"error": f"Network request failed: {str(e)}"}
        return {"error": "All retries failed"}

    def _send_stream_request(self, payload, retries=3, backoff_factor=0.5):
        """
        发送流式请求到API，支持TTFT计算

        Args:
            payload: 请求负载
            retries: 重试次数
            backoff_factor: 退避因子

        Returns:
            (响应内容, 首个令牌时间, 响应时间)
        """
        if payload is None:
            logger.error("Payload cannot be None")
            return "", 0, 0
            
        for attempt in range(retries):
            try:
                start_time = time.time()
                first_token_time = None
                response_chunks = []

                response = self.session.post(
                    url=self.full_url,
                    json=payload,
                    timeout=60,
                    stream=True
                )

                if response is None:
                    logger.error("Received None response")
                    return "", 0, 0

                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            current_time = time.time()
                            if first_token_time is None:
                                first_token_time = current_time - start_time

                            try:
                                line_str = line.decode('utf-8')
                                if line_str.startswith('data: '):
                                    data_str = line_str[6:]
                                    if data_str.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(data_str)
                                        if 'choices' in chunk_data and chunk_data['choices']:
                                            delta = chunk_data['choices'][0].get('delta', {})
                                            content = delta.get('content', '')
                                            if content:
                                                response_chunks.append(str(content))
                                    except json.JSONDecodeError:
                                        continue
                            except UnicodeDecodeError:
                                logger.warning("Failed to decode response line")
                                continue

                    end_time = time.time()
                    full_content = ''.join(response_chunks)
                    response_time = end_time - (start_time + (first_token_time or 0))

                    # 确保返回的字符串不为None
                    return full_content if full_content is not None else "", first_token_time or 0, response_time

                elif response.status_code == 404:
                    alt_path = self._get_alternative_path()
                    self.api_path = alt_path
                    self.full_url = f"{self.base_host}{self.api_path}"
                    continue
                elif response.status_code == 502:
                    if attempt < retries - 1:
                        sleep_time = backoff_factor * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                    else:
                        return "", 0, 0
                else:
                    return "", 0, 0

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(backoff_factor * (2 ** attempt))
                    continue
                logger.error(f"Stream request failed: {str(e)}")
                return "", 0, 0

        return "", 0, 0

    def _get_alternative_path(self):
        """
        获取备选API路径

        Returns:
            备选API路径
        """
        alternatives = ["/v1/chat/completions"]
        for path in alternatives:
            if path != self.api_path:
                return path
        return self.api_path

    def call_model(self, messages, model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stream=False):
        """
        调用语言模型

        Args:
            messages: 消息列表
            model: 模型名称
            max_tokens: 最大生成令牌数
            temperature: 温度参数
            stream: 是否使用流式响应

        Returns:
            模型响应结果，如果stream=True则返回(结果, 首个令牌时间, 响应时间)
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        if stream:
            return self._send_stream_request(payload)
        else:
            result = self._send_request(payload)
            return self._parse_response(result)

    def _parse_response(self, response):
        """
        解析API响应

        Args:
            response: API响应

        Returns:
            解析后的响应内容
        """
        if response is None:
            logger.error("Received None response from API")
            return ""
        if "error" in response:
            return response
        try:
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                return content if content is not None else ""
            elif "response" in response and "text" in response["response"]:
                text = response["response"]["text"]
                return text if text is not None else ""
            elif "result" in response and "output" in response["result"]:
                output = response["result"]["output"]
                return output if output is not None else ""
            elif "text" in response:
                text = response["text"]
                return text if text is not None else ""
            elif "data" in response and "content" in response["data"]:
                content = response["data"]["content"]
                return content if content is not None else ""
            else:
                logger.warning("Unable to parse response structure")
                return ""
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Response parsing error: {str(e)}")
            return ""

    def _fix_truncated_json(self, json_str):
        """
        尝试修复截断的JSON字符串
        主要处理TimeCAP等方法返回的超长数组被截断的情况
        """
        try:
            # 移除首尾空白
            json_str = json_str.strip()
            
            # 检查是否以 { 开头但没有正确结束
            if json_str.startswith('{') and not json_str.endswith('}'):
                # 查找最后一个完整的数值
                # 针对 "Pred_Series": [-1.2943205995599538, -1.2943205995599538, ... -1. 这种情况
                
                # 找到 "Pred_Series": [ 的位置
                pred_series_match = re.search(r'"Pred_Series"\s*:\s*\[', json_str)
                if pred_series_match:
                    start_pos = pred_series_match.end() - 1  # 包含 [
                    
                    # 从开始位置查找数组内容
                    array_content = json_str[start_pos:]
                    
                    # 改进的数值匹配模式，更准确地识别完整数值
                    # 匹配完整的浮点数：可选负号 + 数字 + 可选小数点和小数部分
                    number_pattern = r'-?\d+(?:\.\d+)?'
                    
                    # 查找所有数值，但只保留完整的
                    pos = 1  # 跳过开头的 [
                    complete_numbers = []
                    
                    while pos < len(array_content):
                        # 跳过空白字符和逗号
                        while pos < len(array_content) and array_content[pos] in ' \t\n\r,':
                            pos += 1
                        
                        if pos >= len(array_content):
                            break
                            
                        # 尝试匹配数值
                        match = re.match(number_pattern, array_content[pos:])
                        if match:
                            num_str = match.group()
                            # 检查数值是否被截断（在字符串末尾且没有后续的逗号或括号）
                            end_pos = pos + len(num_str)
                            
                            # 如果这是最后一个数值且可能被截断，跳过它
                            if end_pos >= len(array_content) - 5:  # 给一些缓冲空间
                                # 检查是否有合适的结束符
                                remaining = array_content[end_pos:].strip()
                                if not remaining or remaining[0] not in ',]':
                                    break  # 可能被截断，停止处理
                            
                            try:
                                complete_numbers.append(float(num_str))
                                pos = end_pos
                            except ValueError:
                                pos += 1
                        else:
                            pos += 1
                    
                    if complete_numbers:
                        # 构造修复后的JSON
                        fixed_json = '{\n  "Pred_Series": ' + json.dumps(complete_numbers) + '\n}'
                        
                        # 验证修复后的JSON是否有效
                        json.loads(fixed_json)
                        logger.info(f"Successfully fixed truncated JSON with {len(complete_numbers)} complete numbers")
                        return fixed_json
                
                # 如果无法修复Pred_Series，尝试简单的闭合
                # 移除最后不完整的部分
                lines = json_str.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i].strip()
                    # 如果行以数字和逗号结尾，或者是完整的JSON字段，保留到这里
                    if (line.endswith(',') and re.match(r'.*-?\d+(?:\.\d+)?\s*,', line)) or line.endswith('}'):
                        # 重新构建JSON，添加缺失的闭合括号
                        truncated_json = '\n'.join(lines[:i+1])
                        if not truncated_json.endswith('}'):
                            truncated_json += '\n}'
                        
                        # 验证修复后的JSON
                        try:
                            json.loads(truncated_json)
                            return truncated_json
                        except json.JSONDecodeError:
                            continue
                
                # 最后尝试：简单添加闭合括号
                if json_str.count('{') > json_str.count('}'):
                    missing_braces = json_str.count('{') - json_str.count('}')
                    fixed_json = json_str + '}' * missing_braces
                    try:
                        json.loads(fixed_json)
                        return fixed_json
                    except json.JSONDecodeError:
                        pass
            
            # 如果无法修复，返回原始字符串
            return json_str
            
        except Exception as e:
            logger.warning(f"JSON修复失败: {str(e)}")
            return json_str
    


client = LLMClient()


def encode_image(image_path):
    """
    将图像编码为base64字符串

    Args:
        image_path: 图像文件路径

    Returns:
        base64编码的图像字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def callLLM(query, images=None):
    """
    调用LLM模型，支持文本和图像输入

    Args:
        query: 查询文本
        images: 图像文件路径列表

    Returns:
        模型响应结果、输入令牌数、输出令牌数、首个令牌时间、响应时间、总时间和成本
    """
    try:
        # 检查输入参数
        if query is None:
            logger.error("Received None query input")
            return "Error: Query cannot be None", 0, 0, 0, 0, 0, 0
        
        if not isinstance(query, str):
            logger.error(f"Query must be string, got {type(query)}")
            query = str(query)
        
        if not query.strip():
            logger.error("Query cannot be empty")
            return "Error: Query cannot be empty", 0, 0, 0, 0, 0, 0

        messages = []
        # 只有当方法包含(+v)且images不为空时才处理图像
        # 注意：这里需要从上下文获取method参数，暂时使用全局检查
        if has_vision_support() and images:
            if isinstance(images, list):
                for image_path in images:
                    if image_path and os.path.exists(image_path):
                        try:
                            base64_image = encode_image(image_path)
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            })
                        except Exception as e:
                            logger.warning(f"Failed to encode image {image_path}: {str(e)}")
        
        messages.append({"role": "user", "content": query})

        start_time = time.time()

        # 尝试使用流式请求获取TTFT
        try:
            result, ttft, resp_time = client.call_model(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True
            )
        except Exception as stream_error:
            logger.warning(f"Stream request failed, falling back to regular request: {str(stream_error)}")
            # 如果流式请求失败，回退到常规请求
            result = client.call_model(messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stream=False)
            ttft = 0
            resp_time = 0

        # 确保结果不为None
        if result is None:
            logger.error("Received None result from LLM")
            result = ""
        
        if not isinstance(result, str):
            logger.warning(f"Expected string result, got {type(result)}: {result}")
            result = str(result)

        end_time = time.time()
        total_time = end_time - start_time

        # 计算令牌数和成本
        approx_input_tokens = len(query.split())
        approx_output_tokens = len(result.split()) if isinstance(result, str) else 0
        cost = (approx_input_tokens / 1e6) * 1 + (approx_output_tokens / 1e6) * 2

        return result, approx_input_tokens, approx_output_tokens, ttft, resp_time, total_time, cost

    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        return f"Error: {str(e)}", 0, 0, 0, 0, 0, 0


def format_understand_labels(pred_labels):
    """
    转换 Understand 任务的 Pred_Labels 格式
    将 [['Outlier', 'Sudden Spike'], ['Trend', 'Downward']]
    转换为 ['sudden spike outlier', 'downward trend']
    
    Args:
        pred_labels: 原始的预测标签列表
    
    Returns:
        list: 格式化后的标签列表
    """
    if not pred_labels or not isinstance(pred_labels, list):
        return []
    
    formatted_labels = []
    for item in pred_labels:
        if isinstance(item, list) and len(item) >= 2:
            # 将嵌套列表转换为小写字符串并合并
            # 例如: ['Outlier', 'Sudden Spike'] -> 'sudden spike outlier'
            category = item[0].lower() if isinstance(item[0], str) else str(item[0]).lower()
            pattern = item[1].lower() if isinstance(item[1], str) else str(item[1]).lower()
            # 将pattern放在前面，category放在后面，用空格连接
            formatted_label = f"{pattern} {category}"
            formatted_labels.append(formatted_label)
        elif isinstance(item, str):
            # 如果已经是字符串，直接转换为小写
            formatted_labels.append(item.lower())
        else:
            # 其他情况，转换为字符串并小写
            formatted_labels.append(str(item).lower())
    
    return formatted_labels


def parse_llm_output(output, pred_len=None, representative_data=None, task_type=None, method=None):
    """
    解析LLM输出，提取预测标签、序列和影响分数
    
    Args:
        output: LLM的原始输出
        pred_len: 预测长度
        representative_data: 代表性数据
        task_type: 任务类型 (UNDERSTANDING/FORECASTING_NUM/REASONING)
        method: 方法名称
    
    Returns:
        tuple: (pred_labels, pred_series_json, impact_scores_json)
    """

    try:
        # 如果没有提供pred_len，使用PreLen的第一个值
        if pred_len is None:
            pred_len = PreLen[0] if isinstance(PreLen, list) else PreLen

        # 确保输入是字符串类型，处理None值情况
        if output is None:
            # logger.error("Received None input instead of string")  # 静默处理，避免中断进度条
            return [], "[]", "[]"
        if not isinstance(output, str):
            logger.error(f"Expected string input but got {type(output)}")
            return [], "[]", "[]"

        # 检查输入是否为空或只包含空白字符
        if not output or not output.strip():
            logger.error("Empty or whitespace-only input received")
            return [], "[]", "[]"

        # 清理输出格式
        cleaned_output = re.sub(r'^\s*```(?:json)?\s*', '', output, flags=re.IGNORECASE)
        cleaned_output = re.sub(r'\s*```\s*$', '', cleaned_output)
        
        # 修复Python元组格式为JSON数组格式
        # 将 ("key", "value") 转换为 ["key", "value"]
        cleaned_output = re.sub(r'\(([^)]+)\)', r'[\1]', cleaned_output)

        # 新的处理方式：提取各个字段的原始值，然后手动构建response_data
        response_data = {}

        def extract_nested_array(text, field_name):
            """提取嵌套数组字段"""
            # 匹配字段名后的数组内容
            pattern = rf'"{field_name}"\s*:\s*(\[.*?\])'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    array_str = match.group(1)
                    # 预处理数组字符串，处理各种NaN值
                    for pattern, replacement in nan_patterns:
                        array_str = re.sub(pattern, replacement, array_str, flags=re.IGNORECASE)
                    return json.loads(array_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse {field_name} array: {array_str[:100]}... Error: {e}")
                    # 尝试更宽松的解析
                    try:
                        # 移除可能的无效字符并重试
                        cleaned_str = re.sub(r'[^\[\],\d\.\-\s]', '', array_str)
                        return json.loads(cleaned_str) if cleaned_str.strip() else []
                    except:
                        return []
            return []

        # 尝试直接解析JSON
        try:
            # 增强的JSON预处理，处理各种NaN值和格式问题
            preprocessed_output = cleaned_output
            
            # 处理各种形式的NaN值
            nan_patterns = [
                (r'\bnan\b', 'null'),
                (r'\bNaN\b', 'null'), 
                (r'\bNAN\b', 'null'),
                (r'\bNone\b', 'null'),
                (r'\bundefined\b', 'null'),
                (r'\binfinity\b', 'null'),
                (r'\bInfinity\b', 'null'),
                (r'\b-infinity\b', 'null'),
                (r'\b-Infinity\b', 'null')
            ]
            
            for pattern, replacement in nan_patterns:
                preprocessed_output = re.sub(pattern, replacement, preprocessed_output, flags=re.IGNORECASE)
            
            # 处理可能的格式问题
            # 修复缺失的引号
            preprocessed_output = re.sub(r'(\w+):', r'"\1":', preprocessed_output)
            # 修复单引号为双引号
            preprocessed_output = preprocessed_output.replace("'", '"')
            # 移除尾随逗号
            preprocessed_output = re.sub(r',(\s*[}\]])', r'\1', preprocessed_output)
            
            parsed_json = json.loads(preprocessed_output)
            # 如果解析结果是列表，说明输入是纯数组，需要包装成字典
            if isinstance(parsed_json, list):
                # 根据任务类型判断这是什么类型的数据
                if task_type == "FORECASTING_NUM":
                    response_data = {"Pred_Series": parsed_json}
                elif task_type == "UNDERSTANDING":
                    response_data = {"Pred_Labels": parsed_json}
                elif task_type == "FORECASTING_EVENT":
                    response_data = {"Pred_Labels": parsed_json}
                elif task_type == "REASONING":
                    response_data = {"Impact_Scores": parsed_json}
                else:
                    response_data = {"Pred_Series": parsed_json}  # 默认作为预测序列
            else:
                response_data = parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {str(e)}")
            
            # 尝试修复截断的JSON
            try:
                fixed_json = client._fix_truncated_json(cleaned_output)
                if fixed_json != cleaned_output:
                    # 对修复后的JSON进行增强的NaN预处理
                    for pattern, replacement in nan_patterns:
                        fixed_json = re.sub(pattern, replacement, fixed_json, flags=re.IGNORECASE)
                    
                    response_data = json.loads(fixed_json)
                    logger.info("Successfully parsed fixed JSON")
                else:
                    raise json.JSONDecodeError("No fix applied", "", 0)
            except json.JSONDecodeError:
                logger.warning("Fixed JSON still invalid, trying field extraction")

        # 如果JSON解析失败，尝试手动提取字段
        if not response_data:
            logger.info("Attempting manual field extraction...")
            
            # 提取Pred_Labels
            pred_labels_match = re.search(r'"Pred_Labels"\s*:\s*(\[.*?\])', cleaned_output, re.DOTALL)
            if pred_labels_match:
                try:
                    response_data["Pred_Labels"] = json.loads(pred_labels_match.group(1))
                except json.JSONDecodeError:
                    pass

            # 提取Pred_Series
            response_data["Pred_Series"] = extract_nested_array(cleaned_output, "Pred_Series")
            
            # 提取Impact_Scores
            response_data["Impact_Scores"] = extract_nested_array(cleaned_output, "Impact_Scores")

        # 提取数据
        pred_labels = response_data.get("Pred_Labels", [])
        pred_series = response_data.get("Pred_Series", [])
        impact_scores = response_data.get("Impact_Scores", [])

        # 数据验证和处理
        if not isinstance(pred_labels, list):
            pred_labels = []
        if not isinstance(pred_series, list):
            pred_series = []
        if not isinstance(impact_scores, list):
            impact_scores = []

        # 根据任务类型和方法进行特定处理
        if task_type == "FORECASTING_NUM":
            if method in OUR_Method:
                # StreamTS-Agents, StreamTS-Agents (+v) 在预测任务中应该存储 Impact_Scores 到 predict_summary.csv
                # 对于FORECASTING_NUM任务，返回预测序列和影响分数，不返回标签
                return [], json.dumps(pred_series), json.dumps(impact_scores)
            elif method in BASELINE_FORECASTING_NUM:
                # PromptCast, TimeCP 在预测任务中应该存储 Pred_Series
                return [], json.dumps(pred_series), json.dumps([])
        elif task_type == "FORECASTING_EVENT":
            # 对于FORECASTING_EVENT任务，需要返回Pred_Labels以存储到CSV文件中
            formatted_pred_labels = format_understand_labels(pred_labels)
            if method in OUR_Method:
                # StreamTS-Agents, StreamTS-Agents (+v) 在事件预测任务中应该存储 Pred_Labels 和 Impact_Scores
                # 根据新需求，FORECASTING_EVENT任务不存储Pred_Series
                return formatted_pred_labels, json.dumps([]), json.dumps(impact_scores)
            elif method in BASELINE_FORECASTING_EVENT:
                # TimeCAP 在事件预测任务中应该存储 Pred_Labels，但不存储Pred_Series
                return formatted_pred_labels, json.dumps([]), json.dumps([])
        elif task_type == "UNDERSTANDING":
            # 对 UNDERSTANDING 任务的 Pred_Labels 进行格式转换
            formatted_pred_labels = format_understand_labels(pred_labels)
            
            if method in BASELINE_UNDERSTANDING:
                # Window, Window (+v), Inf-LLM, Inf-LLM (+v) 在理解任务中应该存储 Pred_Labels
                # 对于Understanding任务，不应该存储Pred_Series，返回空的pred_series
                return formatted_pred_labels, json.dumps([]), json.dumps([])
            elif method in OUR_Method:
                # StreamTS-Agents, StreamTS-Agents (+v) 在理解任务中应该存储 Pred_Labels 和 Impact_Scores
                # 对于Understanding任务，不应该存储Pred_Series，返回空的pred_series
                return formatted_pred_labels, json.dumps([]), json.dumps(impact_scores)
        elif task_type == "REASONING":
            if method in BASELINE_REASONING:
                # Window, Window (+v), Inf-LLM, Inf-LLM (+v) 在推理任务中应该存储 Pred_Labels
                # 去除"unknown:"前缀的标签
                cleaned_labels = []
                for label in pred_labels:
                    if isinstance(label, str) and label.startswith("unknown:"):
                        cleaned_labels.append(label[8:])  # 移除"unknown:"前缀
                    else:
                        cleaned_labels.append(label)
                return cleaned_labels, json.dumps([]), json.dumps([])
            elif method in OUR_Method:
                # StreamTS-Agents, StreamTS-Agents (+v) 在推理任务中应该存储 Pred_Labels 和 Impact_Scores
                # 去除"unknown:"前缀的标签
                cleaned_labels = []
                for label in pred_labels:
                    if isinstance(label, str) and label.startswith("unknown:"):
                        cleaned_labels.append(label[8:])  # 移除"unknown:"前缀
                    else:
                        cleaned_labels.append(label)
                return cleaned_labels, json.dumps([]), json.dumps(impact_scores)

        # 默认返回（向后兼容）
        return pred_labels, json.dumps(pred_series), json.dumps(impact_scores)

    except Exception as e:
        logger.error(f"JSON parsed error: {str(e)}")
        return [], "[]", "[]"