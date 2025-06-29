"""評測策略模組 - 定義各種從 LLM 輸出中提取答案的策略

包含多種策略：
- PatternMatchingStrategy: 使用正則表達式模式匹配
- BoxExtractionStrategy: 提取 LaTeX 格式的 \\box{} 或 \\boxed{} 中的答案
- CustomRegexStrategy: 使用自定義正則表達式
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class EvaluationStrategy(ABC):
    """評測策略抽象基本類別

    所有評測策略都必須從這個類別繼承，並實現必要的抽象方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer from LLM output."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass

    def validate_output(self, llm_output: Optional[str]) -> bool:
        """Validate the LLM output format."""
        return isinstance(llm_output, str) and llm_output.strip() != ""


class PatternMatchingStrategy(EvaluationStrategy):
    """模式匹配策略 - 使用正則表達式在 LLM 輸出中尋找答案

    預設包含了多種中文和英文的答案模式，能夠處理大部分常見的答案格式
    """

    # 預設的答案匹配模式，包含中英文各種常見格式
    DEFAULT_PATTERNS = [
        r"correct answer is:\n\n\n([A-D]).",
        r"correct answer is:\n\n([A-D]).",
        r"correct answer is:\n([A-D]).",
        r"正確的答案應該是:.*?\b([A-D])\b",
        r"正确的答案应该是:.*?\b([A-D])\b",
        r"正確的選項應為:.*?\b([A-D])\b",
        r"正确的选项应为:.*?\b([A-D])\b",
        r"正確的答案是（([A-D])）",
        r"正确的答案是（([A-D])）",
        r"答案應該是:\s?選?項?\s?([A-D])",
        r"答案应该是:\s?选?项?\s?([A-D])",
        r"答案是:\s?選?項?\s?([A-D])",
        r"答案是:\s?选?项?\s?([A-D])",
        r"答案應為:\s?選?項?\s?([A-D])",
        r"答案应为:\s?选?项?\s?([A-D])",
        r"答案為:\s?([A-D])",
        r"答案应为：\s?([A-D])",
        r"答案為：\s?([A-D])",
        r"答案應該是:\s?([A-D])",
        r"正確答案為 \*\*([A-D])",
        r"正確答案為\(([A-D])\)",
        r"答案應為:\s?([A-D])",
        r"答案应为:\s?([A-D])",
        r"答案是 \*\*([A-D])",
        r"答案 ([A-D]) 正確",
        r"選項 ([A-D]) 正確",
        r"所以答案為([A-D])",
        r"答案：\(([A-D])\)",
        r"答案:\s?([A-D])",
        r"答案：\s?([A-D])",
        r"答案: ([A-D]) ",
        r"答案([A-D]) ",
        r"^選項([A-D])",
        r"^选项([A-D])",
        r"^選([A-D])",
        r"^选([A-D])",
        r"([A-D]). ",
        r"([A-D]).",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)

    def get_strategy_name(self) -> str:
        return "pattern"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer using regex patterns."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None

    def add_pattern(self, pattern: str):
        """Add a custom pattern to the strategy."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)


class BoxExtractionStrategy(EvaluationStrategy):
    """Strategy that extracts answers from LaTeX-style box formatting."""

    DEFAULT_PATTERNS = [r"\\{1,2}box{([A-D])}", r"\\{1,2}boxed{([A-D])}"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)

    def get_strategy_name(self) -> str:
        return "box"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer from box/boxed formatting."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None

    def add_pattern(self, pattern: str):
        """Add a custom box pattern to the strategy."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)


class CustomRegexStrategy(EvaluationStrategy):
    """Strategy that allows custom regex patterns."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not self.config.get("patterns"):
            raise ValueError("CustomRegexStrategy requires 'patterns' in config")
        self.patterns = self.config["patterns"]

    def get_strategy_name(self) -> str:
        return "custom_regex"

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """Extract answer using custom regex patterns."""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None


class EvaluationStrategyFactory:
    """Factory class for creating evaluation strategy instances."""

    _registry: Dict[str, Type[EvaluationStrategy]] = {
        "pattern": PatternMatchingStrategy,
        "box": BoxExtractionStrategy,
        "custom_regex": CustomRegexStrategy,
    }

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[EvaluationStrategy]):
        """Register a new evaluation strategy."""
        cls._registry[name] = strategy_class

    @classmethod
    def create_strategy(
        cls, strategy_type: str, config: Optional[Dict[str, Any]] = None
    ) -> EvaluationStrategy:
        """Create an evaluation strategy instance based on type."""
        if strategy_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported strategy type: {strategy_type}. Available types: {available_types}"
            )

        strategy_class = cls._registry[strategy_type]
        return strategy_class(config)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available strategy types."""
        return list(cls._registry.keys())
