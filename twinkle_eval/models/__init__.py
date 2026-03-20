"""LLM 後端模組。

提供 LLM 抽象類別、OpenAIModel 實作，以及 LLMFactory 工廠。
"""

from twinkle_eval.core.abc import LLM
from .base import LLMFactory
from .openai import OpenAIModel

# 向工廠登錄預設後端
LLMFactory.register_llm("openai", OpenAIModel)

__all__ = [
    "LLM",
    "LLMFactory",
    "OpenAIModel",
]
