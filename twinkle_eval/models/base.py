"""LLM 抽象層與工廠類別。"""

from typing import Any, Dict, List, Type

from twinkle_eval.core.abc import LLM


class LLMFactory:
    """LLM 後端工廠類別。"""

    _registry: Dict[str, Type[LLM]] = {}

    @classmethod
    def register_llm(cls, name: str, llm_class: Type[LLM]) -> None:
        """向工廠登錄一個新的 LLM 後端實作。"""
        cls._registry[name] = llm_class

    @classmethod
    def create_llm(cls, llm_type: str, config: Dict[str, Any]) -> LLM:
        """依類型名稱建立 LLM 實例。"""
        if llm_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"不支援的 LLM 類型: {llm_type}. 可用類型: {available_types}"
            )
        llm_class = cls._registry[llm_type]
        return llm_class(config)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """回傳所有已登錄的 LLM 類型列表。"""
        return list(cls._registry.keys())
