"""
🌟 Twinkle Eval - 高效且準確的 AI 模型評測工具

一個專為 LLM（Large Language Model）設計的評測框架，
採用並行且隨機化測試方法，提供客觀的模型性能分析與穩定性評估。

主要功能：
- 支援多種資料集格式（JSON, JSONL, CSV, TSV, Parquet）
- 並行處理提升評測效率
- 選項隨機排列避免位置偏好
- 多種答案提取策略
- 詳細的統計分析和結果輸出
- 支援多種 LLM API（OpenAI 相容格式）

使用範例：
    from twinkle_eval import TwinkleEvalRunner
    
    runner = TwinkleEvalRunner("config.yaml")
    runner.initialize()
    runner.run_evaluation()

作者：Twinkle AI Team
授權：MIT License
"""

__version__ = "1.1.6"
__author__ = "Twinkle AI Team"
__license__ = "MIT"

from .config import ConfigurationManager, load_config
from .dataset import Dataset, find_all_evaluation_files
from .evaluation_strategies import (
    BoxExtractionStrategy,
    CustomRegexStrategy,
    EvaluationStrategy,
    EvaluationStrategyFactory,
    PatternMatchingStrategy,
)
from .evaluators import Evaluator, RateLimiter
from .exceptions import (
    ConfigurationError,
    DatasetError,
    EvaluationError,
    ExportError,
    LLMError,
    TwinkleEvalError,
    ValidationError,
)

# 匯入主要類別和函數，方便使用者直接從套件層級使用
from .main import TwinkleEvalRunner, create_cli_parser
from .models import LLM, LLMFactory, OpenAIModel

# 定義 __all__ 以控制 from twinkle_eval import * 的行為
__all__ = [
    # 版本資訊
    "__version__",
    "__author__",
    "__license__",
    # 主要類別
    "TwinkleEvalRunner",
    "ConfigurationManager",
    "LLM",
    "LLMFactory",
    "OpenAIModel",
    "Dataset",
    "Evaluator",
    "RateLimiter",
    # 評測策略
    "EvaluationStrategy",
    "PatternMatchingStrategy",
    "BoxExtractionStrategy",
    "CustomRegexStrategy",
    "EvaluationStrategyFactory",
    # 工具函數
    "load_config",
    "find_all_evaluation_files",
    "create_cli_parser",
    # 異常類別
    "TwinkleEvalError",
    "ConfigurationError",
    "LLMError",
    "EvaluationError",
    "DatasetError",
    "ExportError",
    "ValidationError",
]


def get_version():
    """取得 Twinkle Eval 版本號"""
    return __version__


def get_info():
    """取得 Twinkle Eval 套件資訊"""
    return {
        "name": "Twinkle Eval",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "高效且準確的 AI 模型評測工具",
        "url": "https://github.com/ai-twinkle/Eval",
    }
