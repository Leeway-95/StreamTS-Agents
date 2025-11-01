"""
CONFIG
"""
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# 实验任务
# TASK = ["UNDERSTANDING", "REASONING", "FORECASTING_NUM", "FORECASTING_EVENT"]
TASK = ["UNDERSTANDING"]
# 实验方法
OUR_Method = ["StreamTS-Agents"]
BASELINE_UNDERSTANDING = []
# BASELINE_UNDERSTANDING = ["Window", "Inf-LLM"]
BASELINE_REASONING = ["Window", "Inf-LLM"]
BASELINE_FORECASTING_NUM = ["PromptCast", "TimeCP"]
BASELINE_FORECASTING_EVENT = ["TimeCAP"]
# 实验数据集
DATASET_UNDERSTANDING = {"TSQA"}
DATASET_REASONING = {"QATS-4","MCQ2"}
DATASET_FORECASTING_NUM = {"ETTm","Weather","Gold"}
DATASET_FORECASTING_EVENT = {"Weather_ny","Weather_sf","Weather_hs","Finance_sp500","Finance_nikkei","Healthcare_mortality","Healthcare_positive"}

# Preprocess 相关配置 True False
GEN_STREAM_CNT = 10                   # 生成流的数量
TSQA_SAMPLE_RAN_CNT = 10             # TSQA计数
TSQA_PATTERN = False                 # 是否创建TSQA模式图像
TSQA_PATTERN_COMBINE = False         # 是否创建TSQA模式组合图像
OUTPUT_DATASET_IMAGE = False         # 是否输出Dataset图像
Task_TO_PROCESS = ["outlier", "seasonal", "trend", "volatility", "causal", "deductive",
                   "local-inductive", "local-cluster-inductive", "local-correlation-inductive",
                   "shape-cluster-inductive", "shape-correlation-inductive"] # 任务标签
LABELS_PRIORITY_ORDER = [
    # 优先级1: outlier类别
    "sudden spike outlier", "level shift outlier", "outlier",
    # 优先级2: seasonal类别
    "fixed seasonal", "shifting seasonal", "seasonal",
    # 优先级3: trend类别
    "upward trend", "downward trend", "trend",
    # 优先级4: volatility类别
    "increased volatility", "decreased volatility", "volatility",
    # 其他标签
    "causal", "deductive", "local-inductive", "local-cluster-inductive",
    "local-correlation-inductive", "shape-cluster-inductive", "shape-correlation-inductive"
]
DATASET_PATHS = {
    "ETTm": os.path.join(_project_root, "datasets/Ettm/ETTm"),
    "Weather": os.path.join(_project_root, "datasets/Weather/Weather"),
    "Gold": os.path.join(_project_root, "datasets/Gold/Gold"),
    "TSQA": os.path.join(_project_root, "datasets/TSQA/TSQA"),
    "WeatherQA": os.path.join(_project_root, "datasets/WeatherQA/WeatherQA"),
    "AIOps": os.path.join(_project_root, "datasets/AIOps/AIOps"),
    "NAB": os.path.join(_project_root, "datasets/NAB/NAB"),
    "Oracle": os.path.join(_project_root, "datasets/Oracle/Oracle"),
    "MCQ2": os.path.join(_project_root, "datasets/MCQ2/MCQ2"),
    "Weather_ny": os.path.join(_project_root, "datasets/Weather_ny/Weather_ny"),
    "Weather_sf": os.path.join(_project_root, "datasets/Weather_sf/Weather_sf"),
    "Weather_hs": os.path.join(_project_root, "datasets/Weather_hs/Weather_hs"),
    "Finance_sp500": os.path.join(_project_root, "datasets/Finance_sp500/Finance_sp500"),
    "Finance_nikkei": os.path.join(_project_root, "datasets/Finance_nikkei/Finance_nikkei"),
    "Healthcare_mortality": os.path.join(_project_root, "datasets/Healthcare_mortality/Healthcare_mortality"),
    "Healthcare_positive": os.path.join(_project_root, "datasets/Healthcare_positive/Healthcare_positive")
}
DATASET_TO_MERGE = ["AIOps", "NAB", "Oracle", "WeatherQA", "MCQ2"] # 要合并的数据集列表

# model_detector 相关配置
IMAGE_TOP_K = 5                        # 图像的Top-K数量
WINDOW_SIZE = 20                       # 窗口大小
JUMP = 1                               # 跳跃步长
SAMPLE_SIZE = 100                      # 样本大小
PEAKS_HEIGHT = 0.81                    # 峰值高度阈值
P_VALUE = 1e-70                        # P值阈值
KNN_CNT = 3                            # KNN邻居数量
OUTPUT_DETECTION_IMAGE = False         # 是否输出Detection(ClaSP)图像

# agent_reasoner 相关配置
LLM_HOST = "http://gpt-proxy.jd.com"  # LLM服务主机地址
API_PATH = "/v1/chat/completions"      # API路径
LLM_API_KEY = "d3da8d77-ca1f-47f3-8442-9eb561e3d9cc"  # API密钥
MODEL = "gpt-4o-mini"                  # 使用的LLM模型
MAX_TOKENS = 12000                  # 最大Tokens限制
TEMPERATURE = 0.2                   # 温度设置
Prompt_PATHS = {
    "StreamTS-Agents-Understand": os.path.join(_project_root, "agent_reasoner/CoT/Understand.txt"),
    "StreamTS-Agents-Reason": os.path.join(_project_root, "agent_reasoner/CoT/Reason.txt"),
    "StreamTS-Agents-Forecast-Event": os.path.join(_project_root, "agent_reasoner/CoT/PCoT_Forecast_Event.txt"),
    "PromptCast": os.path.join(_project_root, "baselines/time_series_llm/prompt/promptcast/promptcast_prompt.txt"),
    "TimeCP": os.path.join(_project_root, "baselines/time_series_llm/prompt/timecp/timecp_prompt.txt"),
    "TimeCAP": os.path.join(_project_root, "baselines/time_series_llm/prompt/timecap/timecap_prompt.txt"),
    "Inf-LLM-Understand": os.path.join(_project_root, "baselines/text_stream_llm/window_stateful/window_stateful_prompt_understand.txt"),
    "Inf-LLM-Reason": os.path.join(_project_root, "baselines/text_stream_llm/window_stateful/window_stateful_prompt_reason.txt"),
    "Window-Understand": os.path.join(_project_root, "baselines/text_stream_llm/window_stateless/window_stateless_prompt_understand.txt"),
    "Window-Reason": os.path.join(_project_root, "baselines/text_stream_llm/window_stateless/window_stateless_prompt_reason.txt")
}
Memory_Pool_PATH = os.path.join(_project_root, 'agent_reasoner/CoT/memory_pool.json')  # 内存池路径
DEFAULT_MEMORY_POOL_MAX_SIZE = 20 # 内存池REP最大值配置
MEM_TOP_K = 5                          # 内存的Top-K数量
MAX_TOP_HM_COUNT = 5                   # 内存最大HM保留数量
MEMORY_POOL_MAX_ITEMS = 7              # 内存池最大项目数
MODE_FILE = 'FILE'                     # 文件模式
MEMORY_STORAGE_MODE = MODE_FILE        # 内存存储模式
SAVE_MEMORY_POOL = True               # 是否保存内存池
# 预测长度和历史长度
PreLen = [48]
HistLen = [192]
Hist_Pre_ETTm = [(192,48),(288,48),(384,48),(480,48)]
Hist_Pre_Weather = [(288,144),(432,144),(576,144),(720,144)]
Hist_Pre_Gold = [(96,24),(144,24),(192,24),(240,24)]

DATASET_MERGE_PATHS = {
    "QATS-4": os.path.join(_project_root, "datasets/QATS-4/QATS-4")
}
# Postprocess 相关配置
TEXT_SIMILARITY_THRESHOLD = 0.5  # 文本相似度阈值
RESULTS_DIR = "../results"
DATASETS_PATH = "../datasets"
LOGS_DIR = "../logs"
OUTPUT_PREDICT_IMAGE = False           # 是否输出PREDICT图像
LOG_LLM_METRICS_PATH = os.path.join(_project_root, 'output/metrics_call_llm.csv')  # LLM指标日志路径
LOG_EXP_METRICS_PATH = os.path.join(_project_root, 'output/metrics_exp_res.csv')  # 实验结果指标日志路径

# Stream 相关配置
PARALLELISM = 4                        # 并行度
FLINK_JOB_NAME = "StreamTS-Agents"            # Flink作业名称
CHECKPOINT_INTERVAL = 60000            # 检查点间隔（毫秒）