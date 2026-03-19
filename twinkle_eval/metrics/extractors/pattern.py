"""正則表達式模式匹配抽取器。"""

import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


class PatternExtractor(Extractor):
    """使用正則表達式在 LLM 輸出中尋找答案。

    預設包含多種中文和英文的答案模式，能夠處理大部分常見的答案格式。
    使用 [A-Z] 而非硬編碼 A/B/C/D，支援任意數量的選項。
    """

    DEFAULT_PATTERNS: List[str] = [
        r"correct answer is:\n\n\n([A-Z]).",
        r"correct answer is:\n\n([A-Z]).",
        r"correct answer is:\n([A-Z]).",
        r"正確的答案應該是:.*?\b([A-Z])\b",
        r"正确的答案应该是:.*?\b([A-Z])\b",
        r"正確的選項應為:.*?\b([A-Z])\b",
        r"正确的选项应为:.*?\b([A-Z])\b",
        r"正確的答案是（([A-Z])）",
        r"正确的答案是（([A-Z])）",
        r"答案應該是:\s?選?項?\s?([A-Z])",
        r"答案应该是:\s?选?项?\s?([A-Z])",
        r"答案是:\s?選?項?\s?([A-Z])",
        r"答案是:\s?选?项?\s?([A-Z])",
        r"答案應為:\s?選?項?\s?([A-Z])",
        r"答案应为:\s?选?项?\s?([A-Z])",
        r"答案為:\s?([A-Z])",
        r"答案应为：\s?([A-Z])",
        r"答案為：\s?([A-Z])",
        r"答案應該是:\s?([A-Z])",
        r"正確答案為 \*\*([A-Z])",
        r"正確答案為\(([A-Z])\)",
        r"答案應為:\s?([A-Z])",
        r"答案应为:\s?([A-Z])",
        r"答案是 \*\*([A-Z])",
        r"答案 ([A-Z]) 正確",
        r"選項 ([A-Z]) 正確",
        r"所以答案為([A-Z])",
        r"答案：\(([A-Z])\)",
        r"答案:\s?([A-Z])",
        r"答案：\s?([A-Z])",
        r"答案: ([A-Z]) ",
        r"答案([A-Z]) ",
        r"^選項([A-Z])",
        r"^选项([A-Z])",
        r"^選([A-Z])",
        r"^选([A-Z])",
        r"([A-Z]). ",
        r"([A-Z]).",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.patterns: List[str] = self._config.get("patterns", self.DEFAULT_PATTERNS)

    def get_name(self) -> str:
        return "pattern"

    def extract(self, llm_output: str) -> Optional[str]:
        """使用正則表達式模式抽取答案。"""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None

    def add_pattern(self, pattern: str) -> None:
        """新增自訂模式。"""
        if pattern not in self.patterns:
            self.patterns.append(pattern)
