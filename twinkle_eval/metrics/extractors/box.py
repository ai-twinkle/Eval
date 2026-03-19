"""LaTeX \\box{} / \\boxed{} 選項抽取器。"""

import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


class BoxExtractor(Extractor):
    """從 LaTeX 格式的 \\box{} 或 \\boxed{} 中抽取選項答案（單一大寫字母）。"""

    DEFAULT_PATTERNS: List[str] = [
        r"\\{1,2}box{([A-Z])}",
        r"\\{1,2}boxed{([A-Z])}",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.patterns: List[str] = self._config.get("patterns", self.DEFAULT_PATTERNS)

    def get_name(self) -> str:
        return "box"

    def extract(self, llm_output: str) -> Optional[str]:
        """從 \\box{} / \\boxed{} 格式中抽取選項字母。"""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None

    def add_pattern(self, pattern: str) -> None:
        """新增自訂 box 模式。"""
        if pattern not in self.patterns:
            self.patterns.append(pattern)
