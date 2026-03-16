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

    def normalize_answer(self, answer: str) -> str:
        """正規化答案以便比較。預設轉大寫並去除首尾空白。"""
        return answer.strip().upper()

    def is_correct(self, predicted: str, correct: str) -> bool:
        """判斷預測答案是否正確。預設為字串完全比對。"""
        return predicted == correct


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


class MathExtractionStrategy(EvaluationStrategy):
    """數學評測策略 - 提取 \\boxed{} 中的數學答案，並用語意等價判斷是否正確。

    需要安裝額外套件：pip install twinkle-eval[math]
    """

    # 找出所有 \boxed{ 或 \box{ 的起始位置
    _BOXED_START = re.compile(r"\\{1,2}boxed?\{")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            from mathruler.grader import grade_answer  # type: ignore[import]

            self._grade_answer = grade_answer
        except ImportError:
            raise ImportError(
                "數學評測策略需要安裝額外套件，請執行：\n"
                "  pip install twinkle-eval[math]"
            )

    def get_strategy_name(self) -> str:
        return "math"

    @staticmethod
    def _extract_boxed_content(text: str) -> List[str]:
        """用括號計數提取所有 \\boxed{...} 的內容，支援巢狀大括號。"""
        results = []
        for match in re.finditer(r"\\{1,2}boxed?\{", text):
            start = match.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(text[start:i - 1])
        return results

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """從 \\boxed{} 中提取數學答案；若無則嘗試取最後一行非空文字。"""
        if not self.validate_output(llm_output):
            return None

        matches = self._extract_boxed_content(llm_output)
        if matches:
            return matches[-1].strip()

        # fallback：取最後一行非空內容
        lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
        return lines[-1] if lines else None

    def normalize_answer(self, answer: str) -> str:
        """數學答案維持原格式，僅去除首尾空白。"""
        return str(answer).strip()

    def is_correct(self, predicted: str, correct: str) -> bool:
        """用 mathruler 進行語意等價判斷，並補強大小寫與排列差異。"""
        if not predicted or not correct:
            return False
        if self._grade_answer(predicted, correct):
            return True
        return self._post_check_equivalence(predicted, correct)

    def _post_check_equivalence(self, predicted: str, gold: str) -> bool:
        """補強 mathruler 漏網的大小寫 / 逗號分隔順序差異。"""
        normalized_pred = self._normalize_latex_commands(predicted)
        normalized_gold = self._normalize_latex_commands(gold)

        if self._grade_answer(normalized_pred, normalized_gold):
            return True
        return self._compare_as_unordered_list(normalized_pred, normalized_gold)

    def _normalize_latex_commands(self, expr: str) -> str:
        """將 LaTeX 指令轉小寫，例如 \\FRAC -> \\frac。"""
        lowered = re.sub(r"\\[A-Za-z]+", lambda m: m.group(0).lower(), expr)
        lowered = re.sub(r"\\mbox\{([^}]+)\}", lambda m: m.group(1), lowered)
        return lowered.lower()

    def _compare_as_unordered_list(self, predicted: str, gold: str) -> bool:
        """允許逗號分隔解集合忽略順序比對，例如 1,-2 等價 -2,1。"""
        pred_items = self._split_simple_commas(predicted)
        gold_items = self._split_simple_commas(gold)

        if not pred_items or not gold_items or len(pred_items) != len(gold_items):
            return False

        used = [False] * len(gold_items)
        for pred_item in pred_items:
            matched = False
            for idx, gold_item in enumerate(gold_items):
                if used[idx]:
                    continue
                if self._grade_answer(pred_item, gold_item):
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _split_simple_commas(self, expr: str) -> List[str]:
        """拆解簡單逗號分隔元素（跳過含 LaTeX 排版的表達式）。"""
        stripped = expr.strip()
        if "\\\\" in stripped or "\\begin" in stripped or "\\end" in stripped:
            return []
        if len(stripped) >= 2 and stripped[0] in "([{<" and stripped[-1] in ")]}>":
            stripped = stripped[1:-1]
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        return parts if len(parts) > 1 else []


class EvaluationStrategyFactory:
    """Factory class for creating evaluation strategy instances."""

    _registry: Dict[str, Type[EvaluationStrategy]] = {
        "pattern": PatternMatchingStrategy,
        "box": BoxExtractionStrategy,
        "custom_regex": CustomRegexStrategy,
        "math": MathExtractionStrategy,
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
