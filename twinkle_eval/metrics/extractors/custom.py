"""自訂正則表達式抽取器。"""

import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


class CustomRegexExtractor(Extractor):
    """使用自訂正則表達式抽取答案。

    需要在 config 中提供 patterns 列表。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        if not self._config.get("patterns"):
            raise ValueError("CustomRegexExtractor 需要在 config 中提供 'patterns' 列表")
        self.patterns: List[str] = self._config["patterns"]

    def get_name(self) -> str:
        return "custom_regex"

    def extract(self, llm_output: str) -> Optional[str]:
        """使用自訂正則表達式抽取答案。"""
        if not self.validate_output(llm_output):
            return None

        for pattern in self.patterns:
            match = re.search(pattern, llm_output)
            if match:
                return match.group(1).strip()
        return None
